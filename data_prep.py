"""
Script to generate various CSVs and get things ready for the deep learning pipeline 
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from PIL import Image
import os
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
import shutil 
from itertools import combinations
import socket
import random
import pickle
import matplotlib.patheffects as path_effects
import shutil

##=================================================================
##DATA PROCESSING FUNCTIONS
##=================================================================

def constructRandomCSVFolds():
    """
    Create random CSVs that each have classes distributed according to the average of each expert annotator
    """
    # USERS = ['Borys','Dugger', 'Flanagan', 'Kofler', 'McAleese']
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    for i in range(0,5):
        for phase in ["train", "val"]:
            for fold in [0,1,2,3]:
                cored, diffuse, caa = 0,0,0
                for user in USERS:
                    if phase == "train":
                        read_df = pd.read_csv("csvs/phase1/cross_validation/train_duplicate_{}_fold_{}.csv".format(user, fold))
                    if phase == "val":
                        read_df = pd.read_csv("csvs/phase1/cross_validation/val_{}_fold_{}.csv".format(user, fold))
                    cored += sum([1 for element in list(read_df["cored"]) if element > .99])
                    diffuse += sum([1 for element in list(read_df["diffuse"]) if element > .99])
                    caa += sum([1 for element in list(read_df["CAA"]) if element > .99])
                cored = int(cored / float(len(USERS))) #average # of cored per user
                diffuse = int(diffuse / float(len(USERS)))
                caa = int(caa / float(len(USERS)))
                cored_entries = np.concatenate((np.zeros(len(read_df) - cored),  np.ones(cored)))
                diffuse_entries = np.concatenate((np.zeros(len(read_df) - diffuse),  np.ones(diffuse)))
                CAA_entries = np.concatenate((np.zeros(len(read_df) - caa),  np.ones(caa)))
                np.random.shuffle(cored_entries)
                np.random.shuffle(diffuse_entries)
                np.random.shuffle(CAA_entries)
                assert len(cored_entries) == len(diffuse_entries) == len(CAA_entries) == len(read_df)
                images = list(read_df["imagename"])
                names = ["random" for i in range(0, len(read_df))]
                random_df = pd.DataFrame(list(zip(names,images, cored_entries, diffuse_entries, CAA_entries)), columns =['username','imagename', 'cored', 'diffuse', 'CAA'])
                assert len(random_df) == len(read_df)
                ##write both train and val random csvs
                if phase == "train":
                    random_df.to_csv("csvs/phase1/cross_validation/train_duplicate_random{}_fold_{}.csv".format(i, fold))
                if phase == "val":
                    random_df.to_csv("csvs/phase1/cross_validation/val_random{}_fold_{}.csv".format(i, fold))

def constructRandomTestSet():
    """
    Requires expert test sets, will write a random test set abiding by the same avg class distributions for each expert test set
    """
    # USERS = ["Kofler", "Flanagan", "McAleese", "Dugger", "Borys"]
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    summations = [0,0,0] #number of positive annotations in test set over all users, one for each class 
    for user in USERS:
        df = pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(user))
        for index, row in df.iterrows():
            cored, diffuse, CAA = row['cored'], row['diffuse'], row['CAA']
            if cored > .99:
                summations[0] += 1
            if diffuse > .99:
                summations[1] += 1
            if CAA > .99:
                summations[2] += 1
    avg_counts = [(summations[i] / float(len(USERS))) for i in range(0, 3)]
    cored = [1] * int(avg_counts[0]) + [0] * (len(df) - int(avg_counts[0]))
    random.shuffle(cored)
    diffuse = [1] * int(avg_counts[1]) + [0] * (len(df) - int(avg_counts[1]))
    random.shuffle(diffuse)
    CAA = [1] * int(avg_counts[2]) + [0] * (len(df) - int(avg_counts[2]))
    random.shuffle(CAA)
    df['username'] = "random_test"
    df['cored'] = cored
    df['diffuse'] = diffuse
    df['CAA'] = CAA
    df.to_csv("csvs/phase1/test_set/random_test_set.csv")

def extractConsensusLabels():
    """
    from the combined phase 1 labels csv,
    extracts 5 different CSV labels, one for each consensus scheme of agreed by at least 1, by at least 2, ...., by at least 5
    """
    ##first construct maps for each labeler with key:user, key: image name, value:(cored, diffuse, CAA) label
    # users = ["Dugger", "Borys", "Flanagan", "Kofler", "McAleese"]
    USERS = ["NP{}".format(i) for i in range(1,6)]
    mapp = {user : {} for user in USERS} 
    for user in USERS:
        df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        for index, row in df.iterrows():
            image_name = row["imagename"]
            user = row["username"]
            cored = row["cored"]
            diffuse = row["diffuse"]
            CAA = row["CAA"]
            negative = row["negative"]
            flag = row["flag"]
            notsure = row["notsure"]
            mapp[user][image_name] = (cored, diffuse, CAA, negative, flag, notsure)
    ##make sure the set of images for each labeler are identical
    sets = [set(mapp[user].keys()) for user in USERS]
    for set1 in sets:
        for set2 in sets:
            assert set1 == set2
    images = sets[0]
    ##now construct the new csv with the consensus-of-n strategy with all the mapps in place to query
    for n in range(1, 6):
        new_entries = []
        for image in images:
            new_labels = [] ##running list of tuples, one tuple per user 
            for user in USERS: 
                new_labels.append(mapp[user][image])
            sum0, sum1, sum2, sum3, sum4, sum5 = 0, 0, 0, 0, 0, 0
            for i in range(0, len(new_labels)):
                sum0 += new_labels[i][0]
                sum1 += new_labels[i][1]
                sum2 += new_labels[i][2]
                sum3 += new_labels[i][3]
                sum4 += new_labels[i][4]
                sum5 += new_labels[i][5]
            new_entry0, new_entry1, new_entry2, new_entry3, new_entry4, new_entry5,  = 0, 0, 0, 0, 0, 0
            if sum0 >= n:
                new_entry0 = 1
            if sum1 >= n:
                new_entry1 = 1
            if sum2 >= n:
                new_entry2 = 1
            if sum3 >= n:
                new_entry3 = 1
            if sum4 >= n:
                new_entry4 = 1
            if sum5 >= n:
                new_entry5 = 1
            new_entry = (image, "consensus_of_{}".format(n), new_entry0, new_entry1, new_entry2, new_entry3, new_entry4, new_entry5)
            new_entries.append(new_entry)
        df = pd.DataFrame(new_entries, columns =['imagename','username','cored', 'diffuse', 'CAA', 'negative', 'flag', 'notsure'])
        df = df.sort_values(by="imagename")
        df = df[['imagename', 'username', 'cored', 'diffuse', 'CAA', 'negative', 'flag', 'notsure']]
        df.to_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_" + str(n) + ".csv")

def constructFloatLabels():
    """
    Converts each binary label csv into floating point(portion of labeled bounding boxes that lay in the input image)
    Requires image_details.csv generated from blob_detect.py
    Also creates a conglomerate floating point csv with all expert floating point labels combined into one csv: phaseILabels_floating_point.csv
    """
    USERS = ["NP{}".format(i) for i in range(1,6)] + ["UG{}".format(i) for i in [1,2]]
    CONSENSUS = ["consensus_of_{}".format(n) for n in [1,2,3,4,5]]
    ALL = USERS + CONSENSUS 
    for user in ALL:    
        label_csv = "csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user)    
        #load labels csv
        labels = pd.read_csv(label_csv)
        labels['tilename'] = labels['imagename'].str.split('/').str[-1]
        labels['sourcetile'] = labels['tilename'].str.split('_').str[:-1].str.join('_')

        ##csv contains img coords and bbox coords
        image_details = pd.read_csv("csvs/phase1/image_details_phase1.csv") #details ziqi generated 
        # image_details = pd.read_csv("csvs/phase1/image_details.csv") #details I generated 
        image_details['sourcetile'] = image_details['imagename'].str.split('_').str[:-1].str.join('_')

        ##set the imagename as the index
        image_details = image_details.set_index('imagename')
        ##get coordinates of image and bboxes, and append to labels df as new columns
        img_coords = []
        blob_coords = []
        for index, row in labels.iterrows():
            image = image_details.loc[row['tilename']]
            img_box = image['image coordinates (xywh)']
            img_box = img_box[1:-1].split(' ')
            img_box = [int(x) for x in img_box if x]
            img_coords.append(img_box)
            blob_box = image['blob coordinates (xywh)']
            blob_box = blob_box[1:-1].split(' ')
            blob_box = [int(x) for x in blob_box if x]
            blob_coords.append(blob_box)
        grouped_tiles = labels.groupby(['sourcetile'])
        tiles = list(grouped_tiles.groups)
        labels['img_coords'] = img_coords
        labels['blob_coords'] = blob_coords
        ##calculate overlaps and assign floating labels and construct new final dataframe 
        dfs = []
        for tile in tiles:
            images = labels[labels['sourcetile'] == tile]
            overlap = get_overlap(images['img_coords'].tolist(), images['blob_coords'].tolist())
            labs = np.array(images[['cored','diffuse','CAA','negative','flag','notsure']])
            new_label = np.matmul(overlap, labs)
            df = pd.DataFrame()
            df['username'] = images.username
            if "consensus" not in user:
                df['timestamp'] = images.timestamp
            df['sourcetile']=images.sourcetile
            df['imagename'] = [name for name in images.imagename]
            df['cored'] = new_label[:,0]
            df['diffuse'] = new_label[:,1]
            df['CAA'] = new_label[:,2]
            df['negative'] = new_label[:,3]
            df['flag'] = new_label[:,4]
            df['notsure'] = new_label[:,5]
            dfs.append(df)
        new_labels = pd.concat(dfs)
        ##sort by tilename, and write to file
        new_labels['source'] = new_labels['imagename'].str.split('/').str[0]
        new_labels = new_labels.sort_values(by=['imagename'])
        new_labels.to_csv("csvs/phase1/floating_point_labels/{}_floating_point.csv".format(user))
    
    ##combine all individual floating point labels to one csv
    for i in range(0, len(USERS)):
        if i == 0:
            df = pd.read_csv("csvs/phase1/floating_point_labels/" + USERS[i] + "_floating_point.csv")
        else:
            df_user = pd.read_csv("csvs/phase1/floating_point_labels/" + USERS[i] + "_floating_point.csv")
            df = df.append(df_user)
    df = df.sort_values(by=['imagename'])
    df.to_csv("csvs/phase1/floating_point_labels/phaseILabels_floating_point.csv")

##=================================================================
##HELPER FUNCTIONS
##=================================================================

def get_overlap(image_coords=[np.zeros(4)], blob_coords=[np.ones(4)]):
    """
    IMAGE_COORDS - coordinates of images (xywh)
    (array of arrays)
    BLOB_COORDS - tight bounding box of blob (xywh)
    (array of arrays)
    returns: percent of blob contained within image_coords (dan: of shape n x n for n input arrays)
    """
    # Ensure these are np.arrays
    image_coords = np.array(image_coords)
    blob_coords  = np.array(blob_coords)
    blob_areas = blob_coords[:,2] * blob_coords[:,3]
    img_x_max = image_coords[:,0] + image_coords[:,2]
    img_x_min = image_coords[:,0]
    img_y_max = image_coords[:,1] + image_coords[:,3]
    img_y_min = image_coords[:,1]
    img_x_max = img_x_max[:, np.newaxis]
    img_x_min = img_x_min[:, np.newaxis]
    img_y_max = img_y_max[:, np.newaxis]
    img_y_min = img_y_min[:, np.newaxis]
    blob_x_max = blob_coords[:,0] + blob_coords[:,2]
    blob_x_min = blob_coords[:,0]
    blob_y_max = blob_coords[:,1] + blob_coords[:,3]
    blob_y_min = blob_coords[:,1]
    dx = np.minimum(img_x_max, blob_x_max) - np.maximum(img_x_min, blob_x_min)
    dy = np.minimum(img_y_max, blob_y_max) - np.maximum(img_y_min, blob_y_min)
    # everything negative has no meaningful overlap
    dx = np.maximum(0, dx)
    dy = np.maximum(0, dy)
    return (dx * dy) / blob_areas

##=================================================================
##Runner Calls
##=================================================================
constructRandomCSVFolds()
constructRandomTestSet()
extractConsensusLabels()
constructFloatLabels()











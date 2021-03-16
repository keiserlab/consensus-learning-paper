"""
Script containing code relevant for phase 2
"""
from core import * 

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

def constructFloatLabelsPhase2():
    """
    Converts each binary label csv into a more accurate floating point (portion of labeled bounding boxes that lay in the input image)
    Requires image_details_phase2.csv generated from blob_detect.py
    """
    USERS = ["NP{}".format(i) for i in range(1,6)] + ["UG{}".format(i) for i in [1,2]]
    ALL = USERS 
    for user in ALL:    
        label_csv = "csvs/phase2/annotations/phase2_comparison_{}.csv".format(user)    
        #load labels csv
        labels = pd.read_csv(label_csv)
        labels = labels.rename(columns={"tilename":"imagename"})
        ##only want to make df with enrichment images (i.e. no phase1 images)
        labels = labels[labels['prediction_model'] != 'None (label)']
        labels['imagename'] = [x.replace("phase1/blobs/","") for x in labels['imagename']]
        labels['tilename'] = labels['imagename'].str.split('/').str[-1]
        labels['sourcetile'] = labels['tilename'].str.split('_').str[:-1].str.join('_')
        ##csv contains img coords and bbox coords
        image_details = pd.read_csv("csvs/phase2/image_details_phase2.csv") #details ziqi generated 
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
            labs = np.array(images[['cored annotation','diffuse annotation','CAA annotation','negative annotation','flag annotation','notsure annotation']])
            new_label = np.matmul(overlap, labs)
            df = pd.DataFrame()
            df['username'] = images.username
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
        new_labels.to_csv("csvs/phase2/floating_point_labels/{}_floating_point.csv".format(user))

def createFPComparisonDatasets():
    """
    extracts FP labels and replaces phase2_comparison_{}.csv binary annotations with new floating point labels
    This will be the final labeled annotation set that we use for performance 
    """
    USERS = ["NP{}".format(i) for i in range(1,6)] + ["UG{}".format(i) for i in [1,2]]
    for user in USERS:
        fp_df = pd.read_csv("csvs/phase2/floating_point_labels/{}_floating_point.csv".format(user))
        fp_dict =  {}
        for index, row in fp_df.iterrows():
            fp_dict[row['imagename']] = (row['cored'], row['diffuse'], row['CAA'])
        binary_df = pd.read_csv("csvs/phase2/annotations/phase2_comparison_{}.csv".format(user))
        images = binary_df["tilename"]
        old_cored = binary_df["cored annotation"]
        old_diffuse = binary_df["diffuse annotation"]
        old_CAA = binary_df["CAA annotation"]
        new_cored = [fp_dict[images[i]][0] if images[i] in fp_dict else old_cored[i] for i in range(0, len(images))]
        new_cored = [1 if x >.99 else 0 for x in new_cored]
        new_diffuse = [fp_dict[images[i]][1] if images[i] in fp_dict else old_diffuse[i] for i in range(0, len(images))]
        new_diffuse = [1 if x >.99 else 0 for x in new_diffuse]
        new_CAA = [fp_dict[images[i]][2] if images[i] in fp_dict else old_CAA[i] for i in range(0, len(images))]
        new_CAA = [1 if x >.99 else 0 for x in new_CAA]
        binary_df["cored annotation"] = new_cored
        binary_df["diffuse annotation"] = new_diffuse
        binary_df["CAA annotation"] = new_CAA
        binary_df.to_csv("csvs/phase2/final_labels/phase2_comparison_{}.csv".format(user))

def getIntraraterAgreement(include_phase1_annotations=False,repeats="both"):
    """
    requires phase2_comparison_{USER}.csv (the binary label one is fine), calculates average intrarater agreement for each user through 
    1)the kappa statistic
    2) simple accuracy
    looking at images that this user annotated multiple times 
    will pickle the results
    INCLUDE_PHASE1_ANNOTATIONS: whether to include the annotations from phase 1
    REPEATS:
        if REPEATS == "both", will include both self-repeat images and consensus-repeat images
        if REPEATS == "self" will only include self-repeat images
        if REPEATS == "consensus" will only include consensus-repeat images
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    am_types = [0,1,2]
    user_class_kappa_dict = {} #key: user, key: class, value: list of kappa scores
    user_class_accuracy_dict = {}   
    for user in USERS:
        df_phase2 = pd.read_csv("csvs/phase2/annotations/phase2_comparison_{}.csv".format(user))
        if repeats == "both": 
            repeats_df = df_phase2 
        if repeats == "self":
            self_enrichment_df = df_phase2[df_phase2["username"] == user]
            self_enrichment_df = self_enrichment_df[self_enrichment_df["prediction_model"] == "None (label)"]
            repeats_df = self_enrichment_df
        if repeats == "consensus":
            consensus2_enrichment_df = df_phase2[df_phase2["username"] == "consensus_of_2"] 
            consensus2_enrichment_df = consensus2_enrichment_df[consensus2_enrichment_df["prediction_model"] == "None (label)"]
            repeats_df = consensus2_enrichment_df 
        ##make dictionary mapping image name to list of phase2 annotations 
        img_annotations_dict = {}
        for index, row in repeats_df.iterrows():
            img_name = row["tilename"].replace("phase1/blobs/", "") 
            if img_name not in img_annotations_dict:
                img_annotations_dict[img_name] = [(row["cored annotation"], row["diffuse annotation"], row["CAA annotation"])]
            else:
                img_annotations_dict[img_name].append((row["cored annotation"], row["diffuse annotation"], row["CAA annotation"]))
        #if include_phase1_annotations, append phase1 annotations for calculating intrarater agreement
        if include_phase1_annotations:
            if user == "consensus_of_2":
                df_phase1 = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_2.csv")
            else:
                df_phase1 = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
            for index, row in df_phase1.iterrows():
                img_name = row["imagename"] 
                if img_name not in img_annotations_dict:
                    img_annotations_dict[img_name] = [(row["cored"], row["diffuse"], row["CAA"])]
                else:
                    img_annotations_dict[img_name].append((row["cored"], row["diffuse"], row["CAA"]))
        ##filter out images that are NOT multiply annotated
        images = list(img_annotations_dict.keys())
        for key in images:
            if len(img_annotations_dict[key]) == 1:
                del img_annotations_dict[key]
        user_class_kappa_dict[user] = {am_type: [] for am_type in am_types} #key amyloid type, value: list of kappa scores for each image
        user_class_accuracy_dict[user] = {am_type: [] for am_type in am_types}
        ##now iterate over each image, and get an accuracy/kappa score for its annotations
        for am_type in am_types:
            for img in img_annotations_dict:
                labels_list = [x[am_type] for x in img_annotations_dict[img]]
                p_o =  getFrequencyAccuracy(labels_list)
                p_e = getChanceAgreementFromList(labels_list)
                user_class_kappa_dict[user][am_type].append((p_o - p_e) / float(1 - p_e))
                user_class_accuracy_dict[user][am_type].append(p_o)
            ##condense to average after seeing all images
            user_class_kappa_dict[user][am_type] = (np.mean(user_class_kappa_dict[user][am_type]), np.std(user_class_kappa_dict[user][am_type]))
            user_class_accuracy_dict[user][am_type] = (np.mean(user_class_accuracy_dict[user][am_type]), np.std(user_class_accuracy_dict[user][am_type]))
    pickle.dump(user_class_kappa_dict, open("pickles/intrarater_stats_phase2_include_phase1_{}_repeats_{}_kappa.pkl".format(include_phase1_annotations, repeats), "wb"))
    pickle.dump(user_class_accuracy_dict, open("pickles/intrarater_stats_phase2_include_phase1_{}_repeats_{}_accuracy.pkl".format(include_phase1_annotations, repeats), "wb"))

def getAverageIntraRaterAgreement(include_phase1_annotations=False, repeats=None):
    """
    Method to print the average intra-rater agreement of 1) novices and 2) experts
    INCLUDE_PHASE1_ANNOTATIONS: whether to include phase 1 annotations when calculating intra-rater agreement
    REPEATS:
        if REPEATS == "both", will include both self-repeat images and consensus-repeat images
        if REPEATS == "self" will only include self-repeat images
        if REPEATS == "consensus" will only include consensus-repeat images
        """
    dictionary = pickle.load(open("pickles/intrarater_stats_phase2_include_phase1_{}_repeats_{}_accuracy.pkl".format(include_phase1_annotations, repeats), "rb"))
    for amyloid_class in [0,1,2]:
        novices = []
        novice_mini = 100
        experts = []
        expert_mini = 100
        for user in ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']: 
            if "UG" in user:
                novice_mini = min(novice_mini, dictionary[user][amyloid_class][0])
                novices.append(dictionary[user][amyloid_class][0])
            if "NP" in user:
                expert_mini = min(expert_mini, dictionary[user][amyloid_class][0])
                experts.append(dictionary[user][amyloid_class][0])
        print(amyloid_class)
        print("averages: novice: " ,  np.mean(novices), ", experts: ", np.mean(experts))
        print("minimums: novice: ",novice_mini, ", experts: ", expert_mini)

def plotIntraraterAgreement(include_phase1_annotations=False, repeats="both"):
    """
    Plots the intrarater agreement bar charts
    Requires "pickles/intrarater_stats_phase2_include_phase1_{}_kappa.pkl" and "pickles/intrarater_stats_phase2_include_phase1_{}_accuracy.pkl"
    INCLUDE_PHASE1_ANNOTATIONS: whether to include the annotations from phase 1
    REPEATS:
        if repeats == "both", will include both self-repeat images and consensus-repeat images
        if repeats == "self" will only include self-repeat images
        if repeats == "consensus" will only include consensus-repeat images
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    for measure in ["kappa", "accuracy"]:
        if measure == "kappa":
            dictionary = pickle.load(open("pickles/intrarater_stats_phase2_include_phase1_{}_repeats_{}_kappa.pkl".format(include_phase1_annotations, repeats), "rb"))
        else:
            dictionary = pickle.load(open("pickles/intrarater_stats_phase2_include_phase1_{}_repeats_{}_accuracy.pkl".format(include_phase1_annotations, repeats), "rb"))
        cored_list, diffuse_list, CAA_list, cored_std, diffuse_std, CAA_std = [], [], [], [], [], []
        for user in USERS:
            cored_list.append(dictionary[user][0][0])
            cored_std.append(dictionary[user][0][1])
            diffuse_list.append(dictionary[user][1][0])
            diffuse_std.append(dictionary[user][1][1])
            CAA_list.append(dictionary[user][2][0])
            CAA_std.append(dictionary[user][2][1])
        x = np.arange(len(USERS))
        width = .24    
        xlabels = USERS
        xlabels = [x.replace("_", " ") for x in xlabels]
        fig, ax = plt.subplots()
        if include_phase1_annotations:
            phase1_included = "Including"
        else:
            phase1_included = "Excluding"
        if repeats == "both":
            plt.title("Intra-rater Agreement:\nAll Repeats {} Phase-One Annotations".format(phase1_included), y=1.03, fontsize=13)
        if repeats == "self":
            plt.title("Intra-rater Agreement:\nSelf-Repeats {} Phase-One Annotations".format(phase1_included), y=1.03, fontsize=13)
        if repeats == "consensus":
            plt.title("Intra-rater Agreement:\nConsensus-Repeats {} Phase-One Annotations".format(phase1_included), y=1.03, fontsize=13)
        rect1 = ax.bar(x, cored_list, width, yerr=cored_std, capsize=3, error_kw=dict(capthick=1), color="maroon", label="Cored")# label=mapp["individuals"])
        rect2 = ax.bar(x + width, diffuse_list, width, yerr=diffuse_std, capsize=3, error_kw=dict(capthick=1), color="bisque", label="Diffuse")# label=mapp["others"])
        rect3 = ax.bar(x + 2*width, CAA_list, width, yerr=CAA_std, capsize=3, error_kw=dict(capthick=1), color="darkorange", label="CAA")# label=mapp["others"])
        plt.ylim((0.0, 1.17))
        plt.xticks(x, xlabels, fontsize=12)
        ax.set_xlabel("Annotator", fontsize=12)
        ax.set_ylabel("{}".format(measure.capitalize()), fontsize=12)
        y_vals = ax.get_yticks()
        ax.set_yticklabels(['{:,.0%}'.format(y) for y in y_vals], fontsize=10)    
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")
        #Shrink current axis and place legend outside plot top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', fontsize=7, bbox_to_anchor=(1, 1.35))
        plt.gcf().subplots_adjust(bottom=0.24, top=.76)
        plt.savefig("figures/intrarater_stats_phase2_include_phase1_{}_{}_repeats_{}.png".format(include_phase1_annotations, measure, repeats), dpi=300)

def createIndividualModelPredictionsForConsensusImageSet(IMAGE_DIR=None, norm=None):
    """
    for the 3,476 images enriched for by the consensus of 2 model, have the individual annotator models predict on these,
    write csv to individual_prediction_on_consensus_image_set.csv
    IMAGE_DIR: directory where images are located
    NORM: numpy object containing image normalization mean and std stats
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']    
    compare_df = pd.read_csv("csvs/phase2/annotations/phase2_comparison_NP4.csv")
    consensus2_enrichment_df = compare_df[compare_df["username"] == "NA (prediction)"] 
    consensus2_enrichment_df = consensus2_enrichment_df[consensus2_enrichment_df["prediction_model"] == "consensus_of_2"]
    consensus_images = [x.replace("phase1/blobs/", "") for x in consensus2_enrichment_df['tilename']]
    consensus_images = consensus_images
    user_pred_dict = {user: [] for user in USERS} #key: user, value: list of tuple predictions
    for user in USERS:
        model = torch.load("models/model_{}_fold_3_l2.pkl".format(user)).cuda()
        data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])
        j = 0
        for image in consensus_images:
            img_as_img = Image.open(IMAGE_DIR + image)
            img_as_img = data_transforms(img_as_img)
            outputs = model(img_as_img.view(1, 3, 256, 256).cuda())
            predictions = torch.sigmoid(outputs).type(torch.cuda.FloatTensor).tolist()[0]
            predictions = tuple(predictions)
            user_pred_dict[user].append(predictions)
    df = pd.DataFrame(list(zip(consensus_images, user_pred_dict["UG1"], user_pred_dict["UG2"], user_pred_dict["NP1"], user_pred_dict["NP2"], user_pred_dict["NP3"], user_pred_dict["NP4"], user_pred_dict["NP5"])), 
        columns =['tilename', 'UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']) 
    df.to_csv("csvs/phase2/individualModelPredictionsForConsensusImageSet.csv")

def instantiatePhase2Dictionaries():
    """
    instantiate relevant dictionaries for calculating phase 2 AUROC/AUPRC metrics:
        consensus_benchmark_dict.pkl - for using a consensus benchmark, will need a map from image to consensus label (according to consensus of 2, for the images enriched by consensus of 2 model) 
        individual_predictions_on_consensus_dict.pkl - instantiate dictionary that keeps track of individual neuropath model predictions on the image set enriched by consensus
    requires phase2_comparison_{}.csv
    """
    ##consensus_benchmark_dict
    consensus_benchmark_dict = {} #key imagename, value: consensus label tuple (cored, diffuse, CAA) range (0->5)
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']  ##we only want the expert annotators for deciding a consensus!
    for user in USERS:
        compare_df = pd.read_csv("csvs/phase2/final_labels/phase2_comparison_" + user + ".csv")
        consensus2_enrichment_df = compare_df[compare_df["username"] == "NA (prediction)"] 
        consensus2_enrichment_df = consensus2_enrichment_df[consensus2_enrichment_df["prediction_model"] == "consensus_of_2"]
        consensus_images = set(list(consensus2_enrichment_df['tilename']))
        for index,row in consensus2_enrichment_df.iterrows():
            image = row["tilename"].replace("phase1/blobs/", "")
            cored = 1 if row["cored annotation"] > .99 else 0
            diffuse = 1 if row["diffuse annotation"] > .99 else 0
            CAA = 1 if row["CAA annotation"] > .99 else 0
            if image not in consensus_benchmark_dict.keys():      
                consensus_benchmark_dict[image] = {"cored": cored, "diffuse": diffuse, "CAA": CAA} #[cored, diffuse, CAA]
            else:
                consensus_benchmark_dict[image]["cored"] += cored 
                consensus_benchmark_dict[image]["diffuse"] += diffuse 
                consensus_benchmark_dict[image]["CAA"] += CAA
    pickle.dump(consensus_benchmark_dict, open("pickles/consensus_benchmark_dict.pkl", "wb"))

    ##individual_predictions_on_consensus_dict
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']    
    individual_predictions_on_consensus_dict = {user: {img : -1 for img in consensus_images} for user in USERS} #key user: key:image, key: amyloid class, value: prediction 
    df = pd.read_csv("csvs/phase2/individualModelPredictionsForConsensusImageSet.csv")
    for index, row in df.iterrows():
        for user in USERS:
            pred = row[user]
            pred = pred.replace("(", "")
            pred = pred.replace(")", "")
            pred = pred.split(",")
            pred = [float(x) for x in pred]
            mapp = {"cored": pred[0], "diffuse": pred[1], "CAA": pred[2]}
            image = row['tilename']
            individual_predictions_on_consensus_dict[user][image] = mapp
    pickle.dump(individual_predictions_on_consensus_dict, open("pickles/individual_predictions_on_consensus_dict.pkl", "wb"))

def ModelEnrichmentAUPRC_AUROC(toggle="A"):
    """
    requires phase2_comparison_{USER}.csvs, and also phase2/individualModelPredictionsForConsensusImageSet.csv
    draws AUROC and AUPRC plots for the following: ["individ_mod_individ_benchmark", "individ_mod_consensus_benchmark", "consensus_mod_individ_benchmark", "consensus_mod_consensus_benchmark"]
    also individual annotator bar charts for these metrics
    also a differential between consensus model performance and individual model performance
    if toggle A - individ mod individ benchmark comes from image set that individ model enriched for
    if toggle B - individual mod individ benchmark comes from the 3,476 images that consensus of 2 enriched for 
    get AUPRC and AUROC plots for how well the individual expert models and the consensus model can prospectively predict what the annotator will label 
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']    
    ##map from image to consensus label (according to consensus of 2, for the images enriched by consensus of 2 model) 
    consensus_benchmark_dict =  pickle.load(open("pickles/consensus_benchmark_dict.pkl", "rb"))
    
    ##individual neuropath model predictions on the image set enriched by consensus
    individual_predictions_on_consensus_dict = pickle.load(open("pickles/individual_predictions_on_consensus_dict.pkl", "rb"))
    
    ##get performance across each user
    graph_types = ["individ_mod_individ_benchmark", "individ_mod_consensus_benchmark", "consensus_mod_individ_benchmark", "consensus_mod_consensus_benchmark", "novice_mod_individ_benchmark", "novice_mod_consensus_benchmark"]
    class_types = ["cored", "diffuse", "CAA"]
    #maps key: user, key:AUPRC or AUROC, key: graph_type, key: class, to value: tuple of (list(x axis values), list(y axis values)) note that lists are same size as base_x_metric (interpolation will be performed)
    performance_mapp = {u: {metric: {graph_type: {am_class: -1 for am_class in class_types} for graph_type in graph_types } for metric in ["AUPRC", "AUROC"]} for u in USERS} 
    ##ROC: x: 1 - spec, y: sens, PRC: precision, sensitivity (Recall)
    base_x_metric = np.linspace(0, 1, 30) #last arg is how many points to plot when we aggregate  
    for graph_type in graph_types:
        for user in USERS:
            if "novice_mod" in graph_type and user not in ["UG1", "UG2"]:
                continue
            if ("individ_mod" in graph_type and user in ["UG1", "UG2"]) or ("consensus_mod" in graph_type and user in ["UG1", "UG2"]):
                continue
            compare_df = pd.read_csv("csvs/phase2/final_labels/phase2_comparison_" + user + ".csv")
            self_enrichment_df = compare_df[compare_df["username"] == "NA (prediction)"]
            self_enrichment_df = self_enrichment_df[self_enrichment_df["prediction_model"] == user]
            consensus2_enrichment_df = compare_df[compare_df["username"] == "NA (prediction)"] 
            consensus2_enrichment_df = consensus2_enrichment_df[consensus2_enrichment_df["prediction_model"] == "consensus_of_2"]
            if graph_type == "individ_mod_individ_benchmark" or graph_type == "novice_mod_individ_benchmark":
                if toggle == "A":
                    enrichment_df = self_enrichment_df 
                else:
                    enrichment_df = consensus2_enrichment_df
            if graph_type == "consensus_mod_individ_benchmark" or graph_type == "consensus_mod_consensus_benchmark" or graph_type == "individ_mod_consensus_benchmark" or graph_type == "novice_mod_consensus_benchmark":
                enrichment_df = consensus2_enrichment_df
            for class_type in class_types:
                if graph_type == "individ_mod_individ_benchmark" or graph_type == "novice_mod_individ_benchmark":
                    if toggle == "A":
                        predicted_list = list(enrichment_df[class_type])
                    else:
                        predicted_list = [individual_predictions_on_consensus_dict[user][image.replace("phase1/blobs/", "")][class_type] for image in enrichment_df['tilename']]
                    annotation_list = list(enrichment_df[class_type + " annotation"]) 
                if graph_type == "individ_mod_consensus_benchmark" or graph_type == "novice_mod_consensus_benchmark":
                    predicted_list = [individual_predictions_on_consensus_dict[user][image.replace("phase1/blobs/", "")][class_type] for image in enrichment_df['tilename']]
                    annotation_list = [consensus_benchmark_dict[image.replace("phase1/blobs/", "")][class_type] for image in enrichment_df['tilename']]
                    annotation_list = [1 if x >= 2 else 0 for x in annotation_list] ##consensus of 2 
                if graph_type == "consensus_mod_individ_benchmark":
                    predicted_list = list(enrichment_df[class_type])
                    annotation_list = list(enrichment_df[class_type + " annotation"]) 
                if graph_type == "consensus_mod_consensus_benchmark":
                    predicted_list = list(enrichment_df[class_type])
                    annotation_list = [consensus_benchmark_dict[image.replace("phase1/blobs/", "")][class_type] for image in enrichment_df['tilename']]
                    annotation_list = [1 if x >= 2 else 0 for x in annotation_list] ##consensus of 2
                fpr, tpr, t = roc_curve(np.array(annotation_list).ravel(), np.array(predicted_list).ravel())
                tpr = interp(base_x_metric, fpr, tpr)
                tpr[0] = 0.0
                precision, recall, t = precision_recall_curve(np.array(annotation_list).ravel(), np.array(predicted_list).ravel())
                recall = interp(base_x_metric, precision, recall)
                recall[0] = 1.0 
                performance_mapp[user]["AUPRC"][graph_type][class_type] = (base_x_metric, recall)#auprc
                performance_mapp[user]["AUROC"][graph_type][class_type] = (base_x_metric, tpr)# auroc
    pickle.dump(performance_mapp, open("pickles/phase2_performance_mapp.pkl", "wb"))
              
def plotPhase2Performance(toggle="A", separate_amateurs=False, plot_consensus_benchmark=True):
    """
    plots AUROC and AUPRC curves for the phase 2 results
    if TOGGLE A - individ mod individ benchmark comes from image set that individ model enriched for
    if TOGGLE B - individual mod individ benchmark comes from the 3,476 images that consensus of 2 enriched for 
    SEPARATE_AMATEURS: whether to show novice results
    PLOT_CONSENSUS_BENCHMARK: whether to include analyses when we use consensus of 2 as an benchmark 
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']    
    base_x_metric = np.linspace(0, 1, 30) #last arg is how many points to plot when we aggregate  
    performance_mapp = pickle.load(open("pickles/phase2_performance_mapp.pkl", "rb"))
    ##make ROC and PRC plots
    for score_type in ["AUROC", "AUPRC"]:
        for class_type in ["cored", "diffuse", "CAA"]:
            ##first plotAUROC/AUPRC curves for individuals and and consensus of 2 
            fig, ax = plt.subplots()
            color_map = {"UG1": "orange", "NP5": "blue", "UG2": "yellow", "NP4": "green", "NP2": "pink", "NP3":"magenta", "NP1": "purple"}    
            if score_type == "AUROC": #plot x = y line for ROC curve
                ax.plot([0, 1], [0, 1], linestyle = "--", lw=2, color="black", alpha= 0.8)  
            ys_individ_mod_individ_benchmark, ys_individ_mod_consensus_benchmark, ys_consensus_mod_individ_benchmark, ys_consensus_mod_consensus_benchmark, ys_novice_mod_individ_benchmark, ys_novice_mod_consensus_benchmark = [], [], [], [], [], []
            for user in USERS:
                if user != "UG1" and user != "UG2":
                    x,y = performance_mapp[user][score_type]["individ_mod_individ_benchmark"][class_type]
                    ys_individ_mod_individ_benchmark.append(y)
                    x,y = performance_mapp[user][score_type]["individ_mod_consensus_benchmark"][class_type]
                    ys_individ_mod_consensus_benchmark.append(y)
                    x,y = performance_mapp[user][score_type]["consensus_mod_individ_benchmark"][class_type]
                    ys_consensus_mod_individ_benchmark.append(y)
                    x,y = performance_mapp[user][score_type]["consensus_mod_consensus_benchmark"][class_type]
                    ys_consensus_mod_consensus_benchmark.append(y)
                if user == "UG1" or user == "UG2":
                    x,y = performance_mapp[user][score_type]["novice_mod_individ_benchmark"][class_type]
                    ys_novice_mod_individ_benchmark.append(y)
                    x,y = performance_mapp[user][score_type]["novice_mod_consensus_benchmark"][class_type]
                    ys_novice_mod_consensus_benchmark.append(y)
            ys_individ_mod_individ_benchmark, ys_individ_mod_consensus_benchmark, ys_consensus_mod_individ_benchmark, ys_consensus_mod_consensus_benchmark, ys_novice_mod_individ_benchmark, ys_novice_mod_consensus_benchmark = np.array(ys_individ_mod_individ_benchmark), np.array(ys_individ_mod_consensus_benchmark), np.array(ys_consensus_mod_individ_benchmark), np.array(ys_consensus_mod_consensus_benchmark), np.array(ys_novice_mod_individ_benchmark), np.array(ys_novice_mod_consensus_benchmark)
            mean_ys_individ_mod_individ_benchmark, mean_ys_individ_mod_consensus_benchmark, mean_ys_consensus_mod_individ_benchmark, mean_ys_consensus_mod_consensus_benchmark, mean_ys_novice_mod_individ_benchmark, mean_ys_novice_mod_consensus_benchmark = ys_individ_mod_individ_benchmark.mean(axis=0), ys_individ_mod_consensus_benchmark.mean(axis=0), ys_consensus_mod_individ_benchmark.mean(axis=0), ys_consensus_mod_consensus_benchmark.mean(axis=0), ys_novice_mod_individ_benchmark.mean(axis=0), ys_novice_mod_consensus_benchmark.mean(axis=0)
            std_ys_individ_mod_individ_benchmark, std_ys_individ_mod_consensus_benchmark, std_ys_consensus_mod_individ_benchmark, std_ys_consensus_mod_consensus_benchmark, std_ys_novice_mod_individ_benchmark, std_ys_novice_mod_consensus_benchmark  = ys_individ_mod_individ_benchmark.std(axis=0), ys_individ_mod_consensus_benchmark.std(axis=0), ys_consensus_mod_individ_benchmark.std(axis=0), ys_consensus_mod_consensus_benchmark.std(axis=0), ys_novice_mod_individ_benchmark.std(axis=0), ys_novice_mod_consensus_benchmark.std(axis=0)
            auc_individ_mod_individ_benchmark, auc_individ_mod_consensus_benchmark, auc_consensus_mod_individ_benchmark, auc_consensus_mod_consensus_benchmark, auc_novice_mod_individ_benchmark, auc_novice_mod_consensus_benchmark = auc(base_x_metric, mean_ys_individ_mod_individ_benchmark), auc(base_x_metric, mean_ys_individ_mod_consensus_benchmark), auc(base_x_metric, mean_ys_consensus_mod_individ_benchmark), auc(base_x_metric, mean_ys_consensus_mod_consensus_benchmark), auc(base_x_metric, mean_ys_novice_mod_individ_benchmark), auc(base_x_metric, mean_ys_novice_mod_consensus_benchmark) 
            ##fill between 
            props = dict(boxstyle='round', facecolor='white', alpha=0.85)
            if plot_consensus_benchmark:
                ax.plot(base_x_metric, mean_ys_consensus_mod_consensus_benchmark, '-', color="red", label= "AUC = {}".format(str(auc_consensus_mod_consensus_benchmark)[0:4])) 
                ##consensus mode consensus benchmark will not have a std, sample size = 1
                ax.plot(base_x_metric, mean_ys_individ_mod_consensus_benchmark,'-', color="purple", label= "AUC = {}".format(str(auc_individ_mod_consensus_benchmark)[0:4]))  
                plt.fill_between(base_x_metric, mean_ys_individ_mod_consensus_benchmark - std_ys_individ_mod_consensus_benchmark, mean_ys_individ_mod_consensus_benchmark + std_ys_individ_mod_consensus_benchmark, color='purple', alpha=0.05)
            ax.plot(base_x_metric, mean_ys_consensus_mod_individ_benchmark, '-', color="gold", label= "AUC = {}".format(str(auc_consensus_mod_individ_benchmark)[0:4]), zorder=3)                      
            plt.fill_between(base_x_metric, mean_ys_consensus_mod_individ_benchmark - std_ys_consensus_mod_individ_benchmark, mean_ys_consensus_mod_individ_benchmark + std_ys_consensus_mod_individ_benchmark, color='gold', alpha=0.05)
            ax.plot(base_x_metric, mean_ys_individ_mod_individ_benchmark, '-', color="blue", label= "AUC = {}".format(str(auc_individ_mod_individ_benchmark)[0:4]))
            plt.fill_between(base_x_metric, mean_ys_individ_mod_individ_benchmark - std_ys_individ_mod_individ_benchmark, mean_ys_individ_mod_individ_benchmark + std_ys_individ_mod_individ_benchmark, color='blue', alpha=0.05)
            if separate_amateurs:
                if plot_consensus_benchmark:
                    ax.plot(base_x_metric, mean_ys_novice_mod_consensus_benchmark, '-', color="darkolivegreen", label= "Novice Models, Consensus benchmark, AUC = {}".format(str(auc_novice_mod_consensus_benchmark)[0:4]))  
                    plt.fill_between(base_x_metric, mean_ys_novice_mod_consensus_benchmark - std_ys_novice_mod_consensus_benchmark, mean_ys_novice_mod_consensus_benchmark + std_ys_novice_mod_consensus_benchmark, color='darkolivegreen', alpha=0.05)
                ax.plot(base_x_metric, mean_ys_novice_mod_individ_benchmark, '-', color="yellowgreen", label= "Novice Models, Individual benchmarks, AUC = {}".format(str(auc_novice_mod_individ_benchmark)[0:4]))  
                plt.fill_between(base_x_metric, mean_ys_novice_mod_individ_benchmark - std_ys_novice_mod_individ_benchmark, mean_ys_novice_mod_individ_benchmark + std_ys_novice_mod_individ_benchmark, color='yellowgreen', alpha=0.05)
            if score_type == "AUROC":
                ax.set_xlabel("FPR", fontsize=14)
                ax.set_ylabel("TPR", fontsize=14) 
            if score_type == "AUPRC":
                ax.set_xlabel("Precision", fontsize=14)
                ax.set_ylabel("Recall", fontsize=14) 
            plt.ylim(0, 1.0)
            plt.xlim(0, 1.0)
            caps = {"cored": "Cored", "diffuse":"Diffuse", "CAA":"CAA"}    
            if score_type == "AUROC":
                plt.title("Receiver Operating Characteristic: {}".format(caps[class_type]), fontsize=16)
            if score_type == "AUPRC":
                plt.title("Precision Recall: {}".format(caps[class_type]), fontsize=16)
            ##lower left legend
            if score_type == "AUPRC":
                ax.legend(loc='lower left', fontsize=10)
            if score_type == "AUROC":
                ax.legend(loc='lower right', fontsize=10)
            plt.savefig("figures/enrichment_curve_toggle_{}_{}_{}.png".format(toggle, score_type, class_type), dpi=300)
            plt.cla()
            plt.clf()

def plotAverageDifferentials(toggle="A"):
    """
    Plot summary bar chart showing the average differentials between consensus model and individual expert models for each amyloid beta class
    if toggle A - individ mod individ benchmark comes from image set that individ model enriched for
    if toggle B - individual mod individ benchmark comes from the 3,476 images that consensus of 2 enriched for 
    """
    ##only calculate over experts
    PROFESSIONALS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']    
    performance_mapp = pickle.load(open("pickles/phase2_performance_mapp.pkl", "rb"))
    for benchmark_type in ["consensus_benchmark", "individ_benchmark"]:
        differential_dict = {am_type: {metric: [] for metric in ["AUROC", "AUPRC"]} for am_type in ["cored", "diffuse", "CAA"]} #key: amyloid_class, key: AUPRC or AUROC value: list of consensus - individual differences
        for class_type in ["cored", "diffuse", "CAA"]:    
            user_individ_auprc = []
            user_consensus_auprc = []
            user_individ_auroc = []
            user_consensus_auroc = []
            for user in PROFESSIONALS:
                x,y = performance_mapp[user]["AUPRC"]["individ_mod_" + benchmark_type][class_type]
                uiauprc = auc(y, x)
                user_individ_auprc.append(uiauprc)
                x,y = performance_mapp[user]["AUROC"]["individ_mod_" + benchmark_type][class_type]
                uiauroc = auc(x, y)
                user_individ_auroc.append(uiauroc)
                x,y = performance_mapp[user]["AUPRC"]["consensus_mod_" + benchmark_type][class_type]
                ucauprc = auc(y, x)
                user_consensus_auprc.append(ucauprc)
                x,y = performance_mapp[user]["AUROC"]["consensus_mod_" + benchmark_type][class_type]
                ucauroc = auc(x,y)
                user_consensus_auroc.append(ucauroc)
                auprc_differential = ucauprc - uiauprc
                auroc_differential = ucauroc - uiauroc
                if user != "UG1" and user != "UG2":
                    differential_dict[class_type]["AUROC"].append(auroc_differential)
                    differential_dict[class_type]["AUPRC"].append(auprc_differential)
        ##plot differential bar graph
        #condense dictionary to average and std 
        for key1 in differential_dict.keys():
            for key2 in differential_dict[key1].keys():
                differential_dict[key1][key2] = (np.mean(differential_dict[key1][key2]),np.std(differential_dict[key1][key2]))
        fig, ax = plt.subplots()
        x = np.arange(3)
        width = .22

        xlabels = ["Cored", "Diffuse", "CAA"]
        y_auprc = [differential_dict[am_class]["AUPRC"][0] for am_class in ["cored", "diffuse", "CAA"]]
        y_auprc_err = [differential_dict[am_class]["AUPRC"][1] for am_class in ["cored", "diffuse", "CAA"]]
        y_auroc = [differential_dict[am_class]["AUROC"][0] for am_class in ["cored", "diffuse", "CAA"]]
        y_auroc_err = [differential_dict[am_class]["AUROC"][1] for am_class in ["cored", "diffuse", "CAA"]]
        print(benchmark_type)
        print(y_auprc)
        print(y_auprc_err)
        print(y_auroc)
        print(y_auroc_err)
        ax.bar(x, y_auprc, width, yerr=y_auprc_err, capsize=3, color="olivedrab", label="AUPRC")
        ax.bar(x + width, y_auroc, width, yerr=y_auroc_err, capsize=3, color="yellow", label="AUROC")
        if toggle == "A":
            plt.ylim(-.2, .45)
        else:
            plt.ylim(-.10, .12)
        ax.axhline(0, linestyle="--", color='black', lw=.80, alpha=0.8)
        plt.xticks(x, xlabels, fontsize=10)
        benchmark_title = "Consensus" if benchmark_type == "consensus_benchmark" else "Individual"           
        plt.title("Consensus Model Performance Relative to Individual,\n{} Benchmark".format(benchmark_title), fontsize=13, y=1.03)
        ax.set_ylabel("Consensus - Individual", fontsize=12)
        #Shrink current axis and place legend outside plot, top right corner 
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper left', fontsize=10, bbox_to_anchor=(-.03, 1.39))
        plt.gcf().subplots_adjust(left=.18, bottom=0.11, top=.74, right=.82) #default: left = 0.125, right = 0.9, bottom = 0.1, top = 0.9
        plt.savefig("figures/enrichment_differential_toggle_{}_{}.png".format(toggle, benchmark_type), dpi=300) 



















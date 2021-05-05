"""
Unit and integration tests
"""
from core import *
import unittest

shutil.rmtree("outputs/")
os.mkdir("outputs/")

class DataSetTests(unittest.TestCase):
    """
    various tests for our dataset set up 
    """

    def testTrainingTestSplit(self):
        """
        Makes sure images from held out test set are not included in training
        """
        for n in range(1,6):
          test_images = list(pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_{}.csv".format(n))['imagename'])
          training_images = []
          for fold in [0,1,2,3]:
              training_images += list(pd.read_csv("csvs/phase1/cross_validation/train_duplicate_fold_{}_thresholding_{}.csv".format(fold, n)))
          self.assertEqual(len(set(test_images).intersection(set(training_images))), 0)

        USERS = ["NP{}".format(i) for i in range(1,6)]
        for user in USERS:
          test_images = list(pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(user))['imagename'])
          training_images = []
          for fold in [0,1,2,3]:
              training_images += list(pd.read_csv("csvs/phase1/cross_validation/train_duplicate_{}_fold_{}.csv".format(user, fold)))
          self.assertEqual(len(set(test_images).intersection(set(training_images))), 0)

    def testIfTestSetImagesAreIdentical(self):
        """
        Test to see if each annotator has the same images in their test set
        """
        USERS = ["NP{}".format(i) for i in range(1,6)]
        test_images = set(pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_2.csv")['imagename'])
        for user in USERS:
            user_test_images = set(pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(user))['imagename'])
            self.assertEqual(test_images, user_test_images)
        for c in [1,2,3,4,5]:
            consensus_test_images = set(pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_{}.csv".format(c))['imagename'])
            self.assertEqual(test_images, consensus_test_images)

    def testConsensusMadeCorrectly(self):
        """
        Test if consensus annotations are correcly constructed
        """
        consensus = list(range(1,6))
        USERS = ["NP{}".format(i) for i in range(1,6)]
        mapp = {user : {} for user in USERS} 
        for user in USERS:
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
            for index, row in df.iterrows():
                mapp[row["username"]][row["imagename"]] = (row["cored"], row["diffuse"], row["CAA"])
   
        for c in consensus:
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_{}.csv".format(c))
            for index, row in df.iterrows():
                imagename = row["imagename"]
                user_summation = np.array([0,0,0])
                for user in USERS:
                    user_summation += np.array(mapp[user][imagename])
            binary_label = [1 if x >= c else 0 for x in user_summation]
            self.assertEqual(binary_label, [row["cored"], row["diffuse"], row["CAA"]])
    
    def testIfConsensusOfNIsIncludedInKLessThanN(self):
        """
        A consensus-of-n image is by definition also a consensus-of-k image such that k is less than n
        Verify this is true
        """
        all_images = list(pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_2.csv")["imagename"])
        image_dict = {img: {consensus: -1} for consensus in range(1,6) for img in all_images} #key imagename: key: consensus, value: tuple label
        for consensus in range(1,6):
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_{}.csv".format(consensus))
            for index, row in df.iterrows():
                image_dict[row["imagename"]][consensus] = (row["cored"], row["diffuse"], row["CAA"])
        for key in image_dict: 
            for i in range(5, 0, -1):
                label = image_dict[key][i]
                indices_with_pos_annotation = [i for i in range(0, len(label)) if label[i] >= 1]
                for j in range(i-1, 0, -1):
                    label_2 = image_dict[key][j]
                    for index in indices_with_pos_annotation:
                        self.assertTrue(label_2[index] >= 1)

    def testIfFPTheSame(self):
        """
        Test to see if the training, val, and test set FP labels (found in csvs/phase1/cross_validation/) match the FP labels found in csvs/phase1/floating_point_labels/ 
        Also test to see if the duplicated image labels used for training are identical 
        """
        USERS = ["NP{}".format(i) for i in range(1,6)]
        for user in USERS:
            cross_val_dict = {}
            ##establish fp dictionary from ziqi's cross val
            dfs = []
            for fold in [0,1,2,3]:
                df_train = pd.read_csv("csvs/phase1/cross_validation/train_duplicate_{}_fold_{}.csv".format(user, fold))
                dfs.append(df_train)
            df_test = pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(user))
            dfs.append(df_test)
            for df in dfs:
                for index, row in df.iterrows():
                    if row["imagename"] in df.keys():
                        self.assertEqual(cross_val_dict[row["imagename"]], (row["cored"], row["diffuse"], row["CAA"]))
                    else:
                        cross_val_dict[row["imagename"]] = (row["cored"], row["diffuse"], row["CAA"])

            ##now establish dictionary from the fp labels found in csvs/phase1/floating_point_labels/ 
            fp_dict = {}
            df = pd.read_csv("csvs/phase1/floating_point_labels/{}_floating_point.csv".format(user))
            for index, row in df.iterrows():
                fp_dict[row["imagename"]] = (row["cored"], row["diffuse"], row["CAA"])

            ##check if they match
            for key in cross_val_dict:
                for amyloid_class in [0,1,2]:
                    self.assertTrue(abs(fp_dict[key][amyloid_class] - cross_val_dict[key][amyloid_class]) < 0.01)

    def testIfModelEnrichmentPredictionsAreConsistent(self):
        """
        makes sure that model predictions in phase2 CSVs are reproducible
        """
        USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5'] 
        norm = np.load("utils/normalization.npy", allow_pickle=True).item()
        data_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm['mean'], norm['std'])])    
        for user in USERS:
            compare_df = pd.read_csv("csvs/phase2/final_labels/phase2_comparison_{}.csv".format(user))
            ##only want self-enrichment images
            compare_df = compare_df[compare_df["prediction_model"] == user]
            images = compare_df["tilename"]
            images = [x.replace("phase1/blobs/", "") for x in images]
            ##create master key of image: prediction tuple
            key = {}
            for index, row in compare_df.iterrows():
                key[row["tilename"].replace("phase1/blobs/", "")] = (row["cored"], row["diffuse"], row["CAA"])    
            model = torch.load("models/model_{}_fold_3_l2.pkl".format(user)).cuda()
            j = 0
            for image in images:
                j += 1
                if j > 100:
                    break
                img_as_img = Image.open("data/tile_seg/blobs/" + image)
                img_as_img = data_transforms(img_as_img)
                outputs = model(img_as_img.view(1, 3, 256, 256).cuda())
                predictions = torch.sigmoid(outputs).type(torch.cuda.FloatTensor).tolist()[0]
                predictions = np.array(tuple(predictions))
                difference = np.square(predictions - np.array(key[image])).mean(axis=None)
                if difference > .001:
                    raise Exception("predictions do not match up: ", image, predictions, key[image])

class FigureTests(unittest.TestCase):
    """
    various tests for figure generation
    """
    def testVennDiagram(self):
        """
        Test if venn diagram software is working correctly, find overlaps manually and compare to the stated results
        """
        ##results that venn diagram produced for consensus of 1
        venn_results_c1 = [("cored", "NP1", 321), ("cored", "NP2", 249), 
            ("cored", "NP3", 104), ("cored", "NP4", 24),("cored", "NP5", 339), 
            ("diffuse", "NP1", 112),("diffuse", "NP2", 50),("diffuse", "NP3", 18),("diffuse", "NP4", 231), 
            ("diffuse", "NP5", 183),("CAA", "NP1",12),("CAA", "NP2",6),("CAA", "NP3",11), 
            ("CAA", "NP4",39),("CAA", "NP5",39)]
        all_images = list(pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_2.csv")["imagename"])
        ##key imagename, key:amyloid class, value: list of annotations in order of NP1, NP2, ..., NP5
        images_dict = {image: {amyloid_class: [-1,-1,-1,-1,-1] for amyloid_class in ["cored", "diffuse", "CAA"]} for image in all_images} 
        USERS = ["NP{}".format(i) for i in range(1,6)]
        ##fill out the dictionary 
        for user in USERS:
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
            for index, row in df.iterrows():
                user_index = int(user[-1]) - 1
                for amyloid_class in ["cored", "diffuse", "CAA"]:
                    images_dict[row["imagename"]][amyloid_class][user_index] = row[amyloid_class] 
        ##now calculate consensus of 1 for each (amyloid_class, user)
        for amyloid_class in ["cored", "diffuse", "CAA"]:
            for user in USERS:
                user_index = int(user[-1]) - 1
                consensus_of_1 = 0
                for image in all_images:
                    full_list = images_dict[image][amyloid_class]
                    ##if user gave it positive label and everyone else gave it zero 
                    if full_list[user_index] == 1 and full_list.count(1) == 1:
                        consensus_of_1 += 1
                self.assertTrue((amyloid_class, user, consensus_of_1) in venn_results_c1)

    def testInterraterAgreement(self):
        """
        test to make sure average interrater agreement scores are correct
        """
        kappa_dict = pickle.load(open("pickles/phase1_kappa_dict_exclude_novices_True_test_set_only_False.pk", "rb"))
        amyloid_classes = ["cored", "diffuse", "CAA"]
        class_avg_dict = {amyloid_class : [] for amyloid_class in amyloid_classes}
        for u1, u2 in kappa_dict:
            for amyloid_class in kappa_dict[(u1,u2)]:
                if u1 != u2:
                    class_avg_dict[amyloid_class].append(kappa_dict[(u1,u2)][amyloid_class])
        for amyloid_class in amyloid_classes:
            class_avg_dict[amyloid_class] = np.mean(class_avg_dict[amyloid_class])
        self.assertEqual(round(class_avg_dict["cored"], 2), 0.50)
        self.assertEqual(round(class_avg_dict["diffuse"], 2), 0.46)
        self.assertEqual(round(class_avg_dict["CAA"], 2), 0.76)

    def testGrids(self):
        """
        Test to make sure that all grid results have all users present
        """
        user_folds_set = set()
        metrics = ["AUPRC", "AUROC"]
        amyloid_classes = [0,1,2]
        for metric in metrics:
            for amyloid_class in amyloid_classes:
                mapp = pickle.load(open("pickles/mapp_{} mapval_type{}_class_{}_random_ensemble_False_multiple_randoms_False.p".format(metric, "test_set", amyloid_class), "rb"))
                for key in mapp.keys():
                    user_folds_set.add(getUser(key[0]))
        self.assertTrue(set(["UG1", "UG2", "NP1", "NP2", "NP3", "NP4", "NP5"] + ["thresholding_{}".format(i) for i in range(1,6)]).issubset(user_folds_set))

    def testIfAverageConsensusGreaterThanAverageExpert(self):
        """
        test to see if consensus performance is greater than individual-expert performance across different benchmarks, and across amyloid-plaque classes
        """
        metrics = ["AUPRC", "AUROC"]
        amyloid_classes = [0,1,2]
        for metric in metrics:
            for amyloid_class in amyloid_classes:
                mapp = pickle.load(open("pickles/mapp_{} mapval_type{}_class_{}_random_ensemble_False_multiple_randoms_False.p".format(metric, "test_set", amyloid_class), "rb"))
                ##exclude things like ensembles, random models and truths, amateurs, and also evaluate expert annotation benchmarks 
                mapp = {k: mapp[k] for k in mapp.keys() if "ensemble" not in k[0] and "random" not in k[0] and "random" not in k[1]
                    and "UG1" not in k[0] and "UG1" not in k[1]
                    and "UG2" not in k[0] and "UG2" not in k[1]}
                ##all benchmarks
                mapp_a = mapp
                ##individuals-expert benchmarks
                mapp_i = {k: mapp[k] for k in mapp.keys() if getUser(k[1]) in ["NP1", "NP2", "NP3", "NP4", "NP5"]}                    
                ##self benchmarks
                mapp_s = {k: mapp[k] for k in mapp.keys() if getUser(k[1]) == getUser(k[0])}
                ##consensus benchmarks
                mapp_c = {k: mapp[k] for k in mapp.keys() if getUser(k[1]) in ["thresholding_{}".format(i) for i in range(1,6)]}
                for mapp_iter in [mapp_s, mapp_c, mapp_i, mapp_a]:
                    average_c = []
                    average_expert = []
                    for key in mapp_iter.keys():
                        user = getUser(key[0])
                        self.assertTrue(user not in ["UG1", "UG2"])
                        benchmark = getUser(key[1])
                        if mapp_iter == mapp_i:
                            self.assertTrue(benchmark not in ["thresholding_{}".format(i) for i in range(1,6)])
                        if "thresholding" in user:
                            average_c.append(mapp_iter[key][0])
                        else:
                            average_expert.append(mapp_iter[key][0])
                    self.assertTrue(np.mean(np.array(average_c)) > np.mean(np.array(average_expert)))

    def testSaliencyMajorities(self):
        """
        test to make sure that there are more cases of (on novice, off consensus) than (off consensus, on novice)
        for each amyloid class, and for each undergraduate 
        """
        granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
        larger_thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
        all_thresholds = sorted(list(set(granular_thresholds + larger_thresholds)))
        for UG in ["UG1", "UG2"]:
            for amyloid_class in [0,1,2]:
                thresh_dict = pickle.load(open("pickles/CAM_threshold_stats_{}_{}.pkl".format(UG, amyloid_class), "rb"))
                for t in all_thresholds:
                    self.assertTrue(thresh_dict[t]["C"][0] <= thresh_dict[t]["A"][0])

    def testSubsetMajority(self):
        """
        test to make sure that consensus-of-two CAM is a subset of novice CAM more than novice CAM is a subset of consensus-of-two CAM
        """
        granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
        for UG in ["UG1", "UG2"]:
            for amyloid_class in [0,1,2]:
                for t in granular_thresholds:
                    thresh_dict = pickle.load(open("pickles/CAM_subset_dict_{}_{}.pk".format(UG, amyloid_class), "rb"))
                    self.assertTrue(thresh_dict["consensus"][t] >= thresh_dict["amateur"][t])
 
    def testEnsembleSuperiority(self):
        """
        test to make sure that ensembles have better performance for each amyloid class, and both consensus and individual expert
        """
        difference_map = pickle.load(open("pickles/ensemble_superiority_difference_map_random_subnet_False_multiple_subnets_False.pkl", "rb"))
        for amyloid_class in [0,1,2]:
            self.assertTrue(np.mean(difference_map[amyloid_class]["consensus"]) > np.mean(difference_map[amyloid_class]["individual"]))

    def testEnsembleStabilityWithRandoms(self):
        """
        test to make sure that most of the ensemble differences between normal and random-included are close to zero
        """
        pairs = [(True, False), (False, True)]
        for pair in pairs:
            for amyloid_class in [0,1,2]:
                mapp = pickle.load(open("pickles/ensemble_difference_values_{}_random_{}_multiple_{}.pkl".format(amyloid_class, pair[0], pair[1]), "rb"))
                self.assertTrue(False not in [x < 0.11 for x in mapp])
                self.assertTrue(np.mean(mapp) < .03)
    
    def testIntraRaterAgreementHigh(self):
        """
        test if novices have intrarater agreement > 0.88 and experts have intrarater agreement > 0.89
        """
        dictionary = pickle.load(open("pickles/intrarater_stats_phase2_include_phase1_True_repeats_both_accuracy.pkl", "rb"))
        for amyloid_class in [0,1,2]:
            for user in ['UG1', 'UG2']:
                self.assertTrue(dictionary[user][amyloid_class][0] >= 0.88)
            for user in ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']: 
                self.assertTrue(dictionary[user][amyloid_class][0] >= 0.89)

    def testIntraRaterAgreementAverages(self):
        """
        test to make sure average intra-rater reports are correct
        """
        intra_dict = {person : {amyloid_class: [] for amyloid_class in [0,1,2]} for person in ["novices", "experts"]}
        dictionary = pickle.load(open("pickles/intrarater_stats_phase2_include_phase1_True_repeats_both_accuracy.pkl", "rb"))
        for user in dictionary: 
            for amyloid_class in dictionary[user]:
                if "UG" in user:
                    intra_dict["novices"][amyloid_class].append(dictionary[user][amyloid_class][0])
                if "NP" in user:
                    intra_dict["experts"][amyloid_class].append(dictionary[user][amyloid_class][0])
        for key in intra_dict:
            for amyloid_class in [0,1,2]:
                intra_dict[key][amyloid_class] = np.mean(intra_dict[key][amyloid_class])
        self.assertEqual(round(intra_dict["novices"][0], 2), 0.92)
        self.assertEqual(round(intra_dict["novices"][1], 2), 0.90)
        self.assertEqual(round(intra_dict["novices"][2], 2), 0.97)
        self.assertEqual(round(intra_dict["experts"][0], 2), 0.93)
        self.assertEqual(round(intra_dict["experts"][1], 2), 0.92)
        self.assertEqual(round(intra_dict["experts"][2], 2), 0.98)

    def testIfConsensusPerformanceGreaterPhase2(self):
        """
        test if consensus-of-two performance is greater than individual experts for each amyloid class and for each performance metric
        """
        performance_mapp = pickle.load(open("pickles/phase2_performance_mapp.pkl", "rb"))
        for benchmark_type in ["consensus_benchmark", "individ_benchmark"]:
            differential_dict = {am_type: {metric: [] for metric in ["AUROC", "AUPRC"]} for am_type in ["cored", "diffuse", "CAA"]} #key: amyloid_class, key: AUPRC or AUROC value: list of consensus - individual differences
            for class_type in ["diffuse", "CAA"]: ##superior performance only expected for diffuse and CAA classes 
                user_individ_auprc = []
                user_consensus_auprc = []
                for user in ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']: 
                    x,y = performance_mapp[user]["AUPRC"]["individ_mod_" + benchmark_type][class_type]
                    uiauprc = auc(x, y)
                    user_individ_auprc.append(uiauprc)
                    x,y = performance_mapp[user]["AUPRC"]["consensus_mod_" + benchmark_type][class_type]
                    ucauprc = auc(x, y)
                    user_consensus_auprc.append(ucauprc)
                self.assertTrue(np.mean(user_individ_auprc) < np.mean(user_consensus_auprc))

    def testConsensusModelConsensusBenchmarkConsistency(self):
        """
        Consensus model evaluated on consensus benchmark should be the same across all epxert annotation sets
        No variability by definition, test if this is the case. 
        """
        performance_mapp = pickle.load(open("pickles/phase2_performance_mapp.pkl", "rb"))
        for class_type in ["cored", "diffuse", "CAA"]:    
            consensus_mod_consensus_benchmark_x, consensus_mod_consensus_benchmark_y  = performance_mapp["NP1"]["AUPRC"]["consensus_mod_consensus_benchmark"][class_type]
            consensus_mod_consensus_benchmark_auprc = auc(consensus_mod_consensus_benchmark_x, consensus_mod_consensus_benchmark_y)
            for user in ['NP2', 'NP3', 'NP4', 'NP5']:
                x,y = performance_mapp[user]["AUPRC"]["consensus_mod_consensus_benchmark"][class_type]
                auprc = auc(x,y)
                self.assertTrue(consensus_mod_consensus_benchmark_auprc == auprc)

class HelperFunctionTests(unittest.TestCase):
    """
    various tests for helper functions
    """
    def testGetUser(self):
        """
        test the getUser method
        """
        strings = ["wdqwdoNP1__(23901", "", "NP2", "THRESHOLDING1", "THRESHOLDING_2", "UG1_", "model_UG2!@#$$^&()"]
        names = [getUser(x) for x in strings]
        self.assertEqual(names, ["NP1", -1, "NP2", -1, -1, "UG1", "UG2"])

    def testGetAccuracy(self):
        """
        test the method getAccuracy
        """
        entries = [([0,0,0,1,1], [1,0,0,1,1]), ([0,0,0], [1,1,1]), ([1], [1])]
        accuracies = [getAccuracy(x[0], x[1]) for x in entries]
        self.assertEqual(accuracies, [0.80, 0, 1.0])

    def testGetFrequencyAccuracy(self):
        """
        test method getFrequencyAccuracy
        """
        lists = [[1,1,0], [0,0,0], [1,1,1,1,0]]
        accuracies = [getFrequencyAccuracy(l) for l in lists]
        self.assertEqual(accuracies, [2/3, 1.0, 0.80])

unittest.main()





















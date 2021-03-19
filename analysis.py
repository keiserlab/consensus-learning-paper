from core import * 
from phase2 import *
from figure import * 

norm = np.load("utils/normalization.npy", allow_pickle=True).item()
num_workers = 5
DATA_DIR = "/srv/nas/mk1/users/dwong/WSIs/tile_seg/blobs/"
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.1, contrast=0.2,saturation=0.2, hue=0.02),
        transforms.RandomAffine(0, translate=(0.05,0.05), scale=(0.9,1.1), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(norm['mean'], norm['std'])
    ]),
    'dev': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm['mean'], norm['std'])
    ]),
}
weight = torch.FloatTensor([1,1,1]).cuda()
criterion = nn.MultiLabelSoftMarginLoss(weight=weight, size_average=False)

##=============================================
## Generate performance results
##=============================================
grid_test(model_type="all", val_type="test_set", eval_random_ensembles=False, eval_multiple_random=False, data_transforms=data_transforms, criterion=criterion, DATA_DIR=DATA_DIR, num_workers=num_workers) #baseline
grid_test(model_type="all", val_type="test_set", eval_random_ensembles=True, eval_multiple_random=False, data_transforms=data_transforms, criterion=criterion, DATA_DIR=DATA_DIR, num_workers=num_workers) 
grid_test(model_type="all", val_type="test_set", eval_random_ensembles=False, eval_multiple_random=True, data_transforms=data_transforms, criterion=criterion, DATA_DIR=DATA_DIR, num_workers=num_workers) 

##=============================================
## Analysis of annotations 
##=============================================
getPositiveAnnotationDistribution(exclude_zero=True)
getUsersClassCounts()
vennDiagramPhase1()
getInterraterAgreement(exclude_amateurs=True, phase="phase1", test_set_only=False)
plotInterraterAgreement(exclude_amateurs=True, phase="phase1", test_set_only=False)

#=============================================
# Consensus Superiority Analysis
#=============================================
getAverageIndividualOrConsensusMetrics()
getRandomAUPRCBaseline()
##calculate consensus vs individual expert models for all 3 amyloid classes
for amyloid_type in [0,1,2]:
    ##evaluate over test set
    eval_set="test_set" 
    ##load the performance results for the specific class
    mapp = pickle.load(open("pickles/mapp_AUPRC mapval_type{}_class_{}_random_ensemble_False_multiple_randoms_False.p".format(eval_set, amyloid_type), "rb"))
    ##clean map of prefix paths
    new_mapp = {}
    for model,csv in mapp:
        value = mapp[(model,csv)]    
        new_mapp[(model, csv[csv.rfind("/") + 1:])] = value
    mapp = new_mapp
    ##we want to exclude results from random models and random annotation sets
    prohibited_models = ["model_random0l2.pkl"]
    prohibited_truth_spaces = ["val_random0.csv", "random_test_set.csv"]
    mapp = {k:mapp[k] for k in mapp.keys() if k[0] not in prohibited_models and k[1] not in prohibited_truth_spaces}
    ##exclude ensembles for this analysis
    mapp = {k:mapp[k] for k in mapp.keys() if "ensemble" not in k[0]}
    ##compare both model space and truth space
    for comparison_type in ["model", "truth"]:
        # include all consensus types
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=True, exclude_consensus=False, exclude_amateurs=True) #compare diagonals
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=False, exclude_amateurs=True) #compare top half to bottom half, or if comparison type == truth, compare left half to right half
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=True, exclude_amateurs=True) #compare top right corner to bottom right corner, or if comparison type == truth, compare bottom left corner to bottom right corner
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=False, exclude_amateurs=True, consensus_of_2_only=False, exclude_individuals=True) #compare top left corner to bottom left corner, or if comparison type == truth, compare top left corner to top right corner
        ##only consensus of 2
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=True, exclude_consensus=False, exclude_amateurs=True, consensus_of_2_only=True) 
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=False, exclude_amateurs=True, consensus_of_2_only=True) 
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=True, exclude_amateurs=True, consensus_of_2_only=True) 
        compareConsensusVsIndividual(mapp, eval_set=eval_set, amyloid_type=amyloid_type, comparison=comparison_type, diagonal_only=False, exclude_consensus=False, exclude_amateurs=True, consensus_of_2_only=True, exclude_individuals=True) 
##now plot the heat map 
plotConsensusGainsHeatMap(comparison="model", eval_set="test_set", exclude_amateurs=True, consensus_of_2_only=False)
plotConsensusGainsHeatMap(comparison="truth", eval_set="test_set", exclude_amateurs=True, consensus_of_2_only=False)
#plot the consensus of superiority heatmaps
plotConsensusGainsHeatMap(comparison="model", eval_set="test_set", exclude_amateurs=True, consensus_of_2_only=True)
plotConsensusGainsHeatMap(comparison="truth", eval_set="test_set", exclude_amateurs=True, consensus_of_2_only=True)

##=============================================
## Saliency analysis 
##=============================================
# generateNoviceAndConsensusOf2CAMs(IMG_DIR=DATA_DIR, norm=norm, save_dir="/srv/nas/mk1/users/dwong/WSIs/CAM_images/")
for UG in ["UG1", "UG2"]:
    getAmateurAndConsensusOf2Stats(amateur=UG, truncated=False, image_dir="/srv/nas/mk1/users/dwong/WSIs/CAM_images/")
    plotNoviceAndConsensusOf2Stats(amateur=UG)
    for amyloid_class in [0,1,2]:
        getAmateurWithConsensusOf2Difference(amateur=UG, amyloid_class=amyloid_class, truncated=False, image_dir="/srv/nas/mk1/users/dwong/WSIs/CAM_images/")
        plotNoviceWithConsensusOf2Difference(amateur=UG, amyloid_class=amyloid_class)
        getSubsetPercentageConsensusOf2WithAmateur(amateur=UG, amyloid_class=amyloid_class, truncated=False, image_dir="/srv/nas/mk1/users/dwong/WSIs/CAM_images/")
        plotSubsetPercentageConsensusOf2WithAmateur(amateur=UG, amyloid_class=amyloid_class)

##=============================================
## Ensemble analysis 
##=============================================
testEnsembleSuperiority(random_subnet=False, multiple_subnets=False)
plotEnsembleSuperiority(random_subnet=False, multiple_subnets=False)
for amyloid_type in [0,1,2]:
    ##single random subnet 
    compareGrids("AUPRC", amyloid_type, exclude_amateurs=True, eval_set="test_set", single_random=True, multiple_subnets=False)
    ##multiple random subnets
    compareGrids("AUPRC", amyloid_type, exclude_amateurs=True, eval_set="test_set", single_random=False, multiple_subnets=True)
plotEnsembleDiffHistograms(single_random=True, multiple_subnets=False)
plotEnsembleDiffHistograms(single_random=False, multiple_subnets=True)

##=============================================
## Phase2 analysis
##=============================================
constructFloatLabelsPhase2()
createFPComparisonDatasets()
for repeats in ["both", "self", "consensus"]:
    getIntraraterAgreement(include_phase1_annotations=True,repeats=repeats)
    getAverageIntraRaterAgreement(include_phase1_annotations=True, repeats=repeats)
    plotIntraraterAgreement(include_phase1_annotations=True, repeats=repeats)
createIndividualModelPredictionsForConsensusImageSet(IMAGE_DIR=DATA_DIR, norm=norm)
instantiatePhase2Dictionaries()
ModelEnrichmentAUPRC_AUROC(toggle="A")
plotPhase2Performance(toggle="A", separate_amateurs=False, plot_consensus_benchmark=True)
plotAverageDifferentials(toggle="A")

##=============================================
## Supplemental 
##=============================================
getCountsOfNegativeFlagUnsure(phase="phase1")
plotNegativeFlagNotSure(phase="phase1")

getCountsOfNegativeFlagUnsure(phase="phase2")
plotNegativeFlagNotSure(phase="phase2")

getAverageAgreementOfEachUser(exclude_amateurs=True, phase="phase1", test_set_only=False)

for am_type in [0,1,2]:
    mapp = pickle.load(open("pickles/mapp_AUPRC mapval_typetest_set_class_{}_random_ensemble_False_multiple_randoms_False.p".format(am_type), "rb"))
    getAverageSortedAxis(mapp, am_type=am_type, axis="row", exclude_self=True, exclude_consensus=True, exclude_amateurs=True)
    getAverageSortedAxis(mapp, am_type=am_type, axis="column", exclude_self=True, exclude_consensus=True, exclude_amateurs=True)
    correlateInterraterWithAverages(am_type=am_type, eval_set="test_set", exclude_amateurs=True)

stratifyPerformanceByStainType(DATA_DIR=DATA_DIR, data_transforms=data_transforms, num_workers=num_workers)
plotPerformanceByStainType()

##weights of normal ensembles
getEnsembleWeights(eval_random_subnets=False, eval_amateurs=False, eval_multiple=False)
plotEnsembleWeights(eval_random_subnets=False, eval_amateurs=False, eval_multiple=False)

# ##weights of ensembles with random subnet 
getEnsembleWeights(eval_random_subnets=True, eval_amateurs=False, eval_multiple=False)
plotEnsembleWeights(eval_random_subnets=True, eval_amateurs=False, eval_multiple=False)

# ##stain stratification
getStainUserClassCounts(dataset="test")
plotStainUserClassCounts(dataset="test")

compareColorNormalizedTestingToUnNormalized(DATA_DIR=DATA_DIR, RAW_DATA_DIR="/srv/nas/mk1/users/dwong/WSIs/unnormalized_tile_seg/", data_transforms=data_transforms, num_workers=num_workers)
plotColorNormVsUnnormalized()





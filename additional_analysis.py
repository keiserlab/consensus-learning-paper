"""
Additional code for analysis that might be asked during review
"""
from core import * 

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

def transformInvariance():
    """
    for each consensus-of-n, compares original test set predictions with predictions when applying random data transforms
    focuses on CAA positive images
    """
    model = torch.load("models/model_all_fold_3_thresholding_2_l2.pkl").cuda()
    consensus_dict = {consensus : {amyloid_class :[] for amyloid_class in [0,1,2]} for consensus in range(1,6)} ##key consensus-of-n, key: amyloid_class, value: list of length 3, consisting of averages over 100 runs for [total_plus_to_minus, total_minus_to_plus, total_same]
    
    for consensus in range(1,6):
        ##first get CAA positive images according to this consensus
        images = []
        df = pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_{}.csv".format(consensus))
        for index, row in df.iterrows():
            if row["CAA"] > 0.99:
                images.append(row["imagename"])
        print(consensus, len(images))

        ##get original predictions with no transforms
        og_preds = []
        for image in images:
            img = Image.open("/srv/nas/mk1/users/dwong/WSIs/tile_seg/blobs/" + image)
            img = data_transforms['dev'](img)
            outputs =  model(img.view(1,3,256,256).cuda())
            predictions = torch.sigmoid(outputs).type(torch.cuda.FloatTensor).tolist()[0]
            predictions = tuple(predictions)
            og_preds.append(predictions)

        ##get 100 new rounds of predictions with random transforms 
        new_preds = [] ##list containing 100 list of image predictions 
        for i in range(0, 100):
            preds_iter_i = [] ##image predictions for round i out of 100 
            for image in images:
                img = Image.open("/srv/nas/mk1/users/dwong/WSIs/tile_seg/blobs/" + image)
                img = data_transforms['train'](img)
                outputs =  model(img.view(1,3,256,256).cuda())
                predictions = torch.sigmoid(outputs).type(torch.cuda.FloatTensor).tolist()[0]
                predictions = tuple(predictions)
                preds_iter_i.append(predictions)
            new_preds.append(preds_iter_i)

        ##check how many results flipped on average and save to dictionary, do for each amyloid class
        for amyloid_class in [0,1,2]:
            flip_results = [] ##list of flip result lists (100 total): [total_plus_to_minus, total_minus_to_plus, total_same]
            for preds_iter_i in new_preds: ##iterate over the 100 rounds results
                total_plus_to_minus = 0
                total_minus_to_plus = 0 
                total_same = 0
                assert (len(preds_iter_i) == len(og_preds))
                for i in range(0, len(preds_iter_i)): ##iterate over each individual prediction
                    new = preds_iter_i[i]
                    old = og_preds[i]
                    if old[amyloid_class] < 0.5 and new[amyloid_class] >= 0.5:
                        total_minus_to_plus += 1
                    elif old[amyloid_class] >= 0.5 and new[amyloid_class] < 0.5:
                        total_plus_to_minus += 1
                    else:
                        total_same += 1
                assert(total_plus_to_minus + total_minus_to_plus + total_same == len(images))
                flip_results.append([total_plus_to_minus, total_minus_to_plus, total_same])
            ##average over the 100
            flip_results = np.mean(np.array(flip_results), axis=0)
            print("consensus: {}, amyloid_class:{}, average: {}".format(consensus, amyloid_class, flip_results))
            consensus_dict[consensus][amyloid_class] = flip_results

    ##now plot the figures
    amyloid_dict = {0: "Cored", 1:"Diffuse", 2:"CAA"}
    for consensus in range(1,6):
        fig, ax = plt.subplots()
        x = np.arange(3)
        width = .15
        for amyloid_class in [0,1,2]:
            averages = consensus_dict[consensus][amyloid_class]
            ax.bar(x + amyloid_class * width, averages, width=width, label=amyloid_dict[amyloid_class])
        plt.title("Transformation Invariance, Consensus-of-{} CAA Images".format(consensus))
        ax.set_xticks(x)
        ax.set_xticklabels(["+ to -", "- to +", "no change"])
        ax.set_xlabel("Prediction Change", fontsize=10)
        ax.set_ylabel("Average Count over 100 Runs", fontsize=10)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height * 0.85])
        ax.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1, 1.39))
        plt.gcf().subplots_adjust(bottom=0.13, top=.76)
        plt.savefig("outputs/consensus_of_{}_CAA_invariance.png".format(consensus), dpi=300)

transformInvariance()


    




        









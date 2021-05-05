"""
script to train the neural networks, both single nets and ensembles
"""
from core import *

if not os.path.isdir("training_results/"):
    os.mkdir("training_results/")

##command line arguments for training - ensemble, consensus or not, cross val fold, user, ensemble params: use amateur, use random subnet, use multiple random nets, unfreeze amateurs
i = 2
if sys.argv[1] == "ensemble":
	ensemble = True
else:
	ensemble = False
if sys.argv[i] == "consensus":
	consensus = True
else:
	consensus = False
fold = sys.argv[i + 1]
user = sys.argv[i + 2]

if len(sys.argv) >= i + 4:
	use_amateur = True if sys.argv[i + 3] == "use amateur" else False
else:
	use_amateur = False
if len(sys.argv) >= i + 5:
	use_random_subnet = True if sys.argv[i + 4] == "random subnet" else False
else:
	use_random_subnet = False
if len(sys.argv) >= i + 6:
	use_multiple_subnets = True if sys.argv[i + 5] == "multiple subnet" else False
else:
	use_multiple_subnets = False
if len(sys.argv) >= i + 7:
	unfreeze_amateurs = True if sys.argv[i + 6] == "unfreeze amateurs" else False
else:
	unfreeze_amateurs = False

save_flags = ""
if ensemble:
	save_flags += "ensemble_"
if use_random_subnet:
	save_flags += "random_subnet_"
if use_multiple_subnets:
	save_flags += "use_multiple_subnets_"
if use_amateur:
	save_flags += "use_amateur_"
if unfreeze_amateurs:
	save_flags += "unfreeze_amateurs_"

norm = np.load("utils/normalization.npy", allow_pickle=True).item()

prefix = "data/tile_seg/"
num_workers = 16

batch_size = 64
if consensus:
	csv_path = {
	    'train': 'csvs/phase1/cross_validation/train_duplicate_fold_{}_thresholding_{}.csv',
	    'dev': 'csvs/phase1/cross_validation/val_fold_{}_thresholding_{}.csv',
	}
else:
	csv_path = {
	    'train': 'csvs/phase1/cross_validation/train_duplicate_{}_fold_{}.csv',
	    'dev': 'csvs/phase1/cross_validation/val_{}_fold_{}.csv',
	}
DATA_DIR = prefix + 'blobs/'
SAVE_DIR = 'models/'
image_classes = ['cored','diffuse','CAA']

print("    save flags: ", save_flags)
print("    ensemble: ", ensemble)
print("    consensus: ", consensus)
print("    fold: ", fold)
print("    user: ", user)
if consensus:
    print("    csv paths: ", csv_path['train'].format(fold, user), csv_path['dev'].format(fold, user))
else:
    print("    csv paths: ", csv_path['train'].format(user, fold), csv_path['dev'].format(user, fold))

if ensemble:
	print("    use_amateur: ", use_amateur)
	print("    use_random_subnet: ",use_random_subnet) 
	print("    use_multiple_subnets:", use_multiple_subnets)
	print("    unfreeze_amateurs:", unfreeze_amateurs)

#////////////////////////////////////////////////////////////////////////////////////////////

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

##setup loaders
image_datasets = {}
dataloaders = {}
dataset_sizes = {}
if consensus:
	image_datasets[user, fold] = {x: MultilabelDataset(csv_path[x].format(fold, user), DATA_DIR, threshold=0.99, transform=data_transforms[x]) for x in ['train', 'dev']}
else:
	#csv for neuropathologist has user first then fold
	image_datasets[user, fold] = {x: MultilabelDataset(csv_path[x].format(user, fold), DATA_DIR, threshold=0.99, transform=data_transforms[x]) for x in ['train', 'dev']}
dataloaders[user, fold] = {x: torch.utils.data.DataLoader(image_datasets[user, fold][x], batch_size=batch_size,shuffle=True, num_workers=num_workers) for x in ['dev', 'train']}
dataset_sizes[user, fold] = {x: len(image_datasets[user, fold][x]) for x in ['train', 'dev']}

##train
torch.manual_seed(123456789)
weight = torch.FloatTensor([1,1,1])
if ensemble:
	model = instantiateEnsembleModel(equally_weighted=False, use_random_subnet=use_random_subnet,use_amateur=use_amateur,cross_fold=fold,use_multiple_random_subnets=use_multiple_subnets, unfreeze_amateurs=unfreeze_amateurs)
else:
	model = Net()
if torch.cuda.is_available():
    print('gpu')
    weight = weight.cuda()
    model = model.cuda()
criterion = nn.MultiLabelSoftMarginLoss(weight=weight, size_average=False)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.03)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.4)
print("user/thresh: ", user, " fold: ", fold)
best_model = train_model(model, criterion, optimizer, exp_lr_scheduler, user, fold, dataloaders=dataloaders, dataset_sizes=dataset_sizes, num_epochs=60)
if consensus:
	torch.save(best_model, SAVE_DIR + save_flags + 'model_all_fold_{}_thresholding_{}_l2.pkl'.format(fold, user))
else:
	torch.save(best_model, SAVE_DIR + save_flags +'model_{}_fold_{}_l2.pkl'.format(user, fold))


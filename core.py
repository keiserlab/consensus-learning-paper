# -*- coding: utf-8 -*-
"""
Core class and method definitions
Organized as follows:
1) Core class definitions
2) Core method definitions
3) Helper method definitions
"""
import time
import torch
import math
torch.manual_seed(123456789)
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.nn import ReLU
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from decimal import Decimal
from torch.utils.data import Dataset
from PIL import Image
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import sys
import shutil
import os 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, roc_auc_score
from skimage.measure import compare_ssim as ssim
import socket 
import pickle 
from itertools import combinations
import scipy 
import venn
import cv2
import random
from scipy import interp
from scipy.interpolate import interp1d

##=================================================
#1) CORE CLASSES
##=================================================

class MultilabelDataset(Dataset):
    def __init__(self, csv_path, img_path, threshold=0.99, transform=None):
        """
        The dataset that wraps the images and meta information 
        CSV_PATH: path to csv file
        IMG_PATH (string): path to the folder where images are
        THRESHOLD: the float point valued threshold at which we call an image example positive if label > threshold
        TRANSFORM: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        c=torch.Tensor(self.data_info.loc[:,'cored'])
        d=torch.Tensor(self.data_info.loc[:,'diffuse'])
        a=torch.Tensor(self.data_info.loc[:,'CAA'])
        c=c.view(c.shape[0],1)
        d=d.view(d.shape[0],1)
        a=a.view(a.shape[0],1)
        self.raw_labels = torch.cat([c,d,a], dim=1)
        self.labels = (torch.cat([c,d,a], dim=1)>threshold).type(torch.FloatTensor)

    def __getitem__(self, index):
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.labels[index]
        raw_label = self.raw_labels[index]
        # Get image name from the pandas df
        single_image_name = str(self.data_info.loc[index,'imagename'])
        # Open image
        img_as_img = Image.open(self.img_path + single_image_name)
        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        # Return image and the label
        return (img_as_img, single_image_label, raw_label, single_image_name)

    def __len__(self):
        return len(self.data_info.index)

class Net(nn.Module):
    """
    The CNN architecture
    """
    def __init__(self, fc_nodes=512, num_classes=3, dropout=0.5):
        super(Net, self).__init__()
        self.drop = 0.2
        self.features = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                      nn.BatchNorm2d(16),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(16, 32, 3, padding=1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(32, 48, 3, padding=1),
                                      nn.BatchNorm2d(48),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(48, 64, 3, padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(64, 80, 3, padding=1),
                                      nn.BatchNorm2d(80),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),
                                      
                                      nn.Conv2d(80, 96, 3, padding=1),
                                      nn.BatchNorm2d(96),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=2, stride=2),)
        
        self.classifier = nn.Sequential(nn.Linear(96 * 4 * 4, num_classes))
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class CustomizedLinearFunction(torch.autograd.Function):
    """
    Autograd function which masks it's weights by 'mask'. 
    reference: https://github.com/uchida-takumi/CustomizedLinear
    """
    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            # change weight to 0 where mask == 0
            weight = weight * mask
        output = input.mm(weight.t()) ##mm is matrix multiplication
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output
    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require them is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, grad_mask

class EnsembleNet(nn.Module):
    """
    Ensemble net that weights individual constituent models using sparse feed forward connections
    Each consituent model's final layer (each of size 3) is concatenated together into a single layer
    Mask is a 2D matrix that specifies which connections from this concatenated layer of size(# of models x 3) to keep (1), and which connections to delete (0) when wiring to the final 3 neuron layer of the ensemble net
    Each EnsembleNet will have all 7 annotator models as attributes no matter what, BUT we'll selectively choose which models to actually ensemble in the forward definition 
    This way we have the option to forward novice nets if we want to (but in this particular study, we do not)
    Random nets will be models 8 - 12, and instantiated if passed as parameters, else won't be attributes
    """
    def __init__(self, mask, model1, model2, model3, model4, model5, model6, model7,model8=None,model9=None,model10=None,model11=None,model12=None,bias=True,amateur_param=False):
        super(EnsembleNet, self).__init__()
        self.ensemble = True
        self.equally_weighted = False
        self.amateur_param = amateur_param
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()
        # mask weight
        self.weight.data = self.weight.data * self.mask
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        if model8 != None: ##using one random constituent net 
            self.model8 = model8
        else:
            self.model8 = None
        if model9 != None: ##using multiple constituent nets
            self.model9 = model9
            self.model10 = model10
            self.model11 = model11
            self.model12 = model12
        else:
            self.model9, self.model10, self.model11, self.model12 = None, None, None, None
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x6 = self.model6(x)
        x7 = self.model7(x)
        ##if random constituent nets are being used they will be forwarded
        if self.model8 != None: 
            x8 = self.model8(x)
        if self.model9 != None:
            x9 = self.model9(x)
            x10 = self.model10(x)
            x11 = self.model11(x)
            x12 = self.model12(x)

        ##if we're using (forwarding) the two amateur raters in our ensemble 
        if self.amateur_param:
            ##plus one random constituent net
            if self.model8 != None and self.model9 == None:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8), dim=1) 
            ##plus multiple random nets
            elif self.model9 != None:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12), dim=1) 
            ##no random nets
            else:
                x = torch.cat((x1,x2,x3,x4,x5,x6,x7), dim=1)
        ##if we're NOT using (forwarding) the two amateur raters in our ensemble 
        else:
            if self.model8 != None and self.model9 == None:
                x = torch.cat((x2,x4,x5,x6,x7,x8), dim=1)
            elif self.model9 != None:
                x = torch.cat((x2,x4,x5,x6,x7,x8,x9,x10,x11,x12), dim=1)
            else:
                x = torch.cat((x2,x4,x5,x6,x7), dim=1)
        return CustomizedLinearFunction.apply(x, self.weight, self.bias, self.mask)
        
    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
   
class EquallyWeightedEnsembleNet(nn.Module):
    """
    Ensemble that simply forwards each constituent net with equal weighting
    """
    def __init__(self, model1, model2, model3, model4, model5, model6, model7,model8=None,model9=None,model10=None,model11=None,amateur_param=True):
        super(EquallyWeightedEnsembleNet, self).__init__()
        self.ensemble = True
        self.equally_weighted = True
        self.amateur_param = amateur_param
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
        self.model7 = model7
        if model8 != None:
            self.model8 = model8
        else:
            self.model8 = None
        if model9 != None:
            self.model9 = model9
            self.model10 = model10
            self.model11 = model11
            self.model12 = model12
        else:
            self.model9, self.model10, self.model11, self.model12 = None, None, None, None
        self.train_loss_curve = []
        self.dev_loss_curve = []
        self.train_auprc = []
        self.dev_auprc = []

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x3 = self.model3(x)
        x4 = self.model4(x)
        x5 = self.model5(x)
        x6 = self.model6(x)
        x7 = self.model7(x)
        if self.model8 != None:
            x8 = self.model8(x)
        if self.model9 != None:
            x9 = self.model9(x)
            x10 = self.model10(x)
            x11 = self.model11(x)
            x12 = self.model12(x)
        if self.amateur_param:
            if self.model8 != None and self.model9 == None:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7,x8)) 
            elif self.model9 != None:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12)) 
            else:
                x = torch.stack((x1,x2,x3,x4,x5,x6,x7)) 
        else:
            if self.model8 != None and self.model9 == None:
                x = torch.stack((x2,x4,x5,x6,x7,x8))
            elif self.model9 != None:
                x = torch.stack((x2,x4,x5,x6,x7,x8,x9,x10,x11,x12))
            else:
                x = torch.stack((x2,x4,x5,x6,x7)) 
        avg = torch.mean(x, dim=0) 
        return avg 

class CamExtractor():
    """
    Extracts CAM features from the model
    Attribution: https://github.com/utkuozbulak/pytorch-cnn-visualizations    
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.features._modules.items():
        # for module_pos, module in self.model.model1.features._modules.items(): #dan's edit for ensembles
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
        Does a full forward pass on the model
        """
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)
        x = self.model.classifier(x) 
        return conv_output, x

class GradCam():
    """
    Produces CAM
    Attribution: https://github.com/utkuozbulak/pytorch-cnn-visualizations    
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Zero grads
        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.numpy()[0]
        # Get convolution outputs
        target = conv_output.data.numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = cv2.resize(cam, (256, 256))
        cam = np.maximum(cam, 0)
        return cam

class GuidedBackprop():
    """
    Produces the gradients generated with guided back propagation from the given image
    Attribution: https://github.com/utkuozbulak/pytorch-cnn-visualizations    
    """
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        for pos, module in self.model.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_hook_function)

    def generate_gradients(self, input_image, target_class):
        # Forward pass
        model_output = self.model(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        gradients_as_arr = self.gradients.data.numpy()[0]
        return gradients_as_arr

##==================================================================
#2)CORE METHOD DEFINITIONS
##==================================================================


def grid_test(model_type="single_net", val_type="val_set", eval_random_ensembles=False, eval_multiple_random=False, data_transforms=None, criterion=None, DATA_DIR=None, num_workers=None):
    """
    Test each model against each csv dataset (essentially constructing a square grid of different model performances on different annotation test sets)
    Saves results as pickle file
    Subset by different flags as follows:
    if MODEL_TYPE = "single_net", only non-ensemble models will be evaluated
    if EVAL_RANDOM_ENSEMBLES, ensemble models using a single random subnet constituent will be evaluated
    if EVAL_MULTIPLE_RANDOM, ensemble models using multiple random subnets will be evaluated
    DATA_TRANSFORMS is a dictionary specifying how to preprocess the image, key: phase (either "train" or "dev", value: transforms.Compose object
    CRITERION specifies the loss function to evaluate
    DATA_DIR is the directory to read images
    NUM_WORKERS is the number of CPU cores to use for computing
    """
    USERS_FOR_MODELS = ['UG1','UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'random0'] + \
            ["thresholding_" + str(i) for i in range(1,6)]
    ##get the models to use
    models_dir = "models/"
    models = os.listdir(models_dir)
    ##first filter out models based on the arguments
    if model_type == "single_net":
        models = [x for x in models if "ensemble" not in x] 
    ##filter out ensembles based on the arguments
    ensembles = [x for x in models if "ensemble" in x]
    individuals = [x for x in models if "ensemble" not in x]
    if eval_random_ensembles:
        ensembles = [x for x in ensembles if "random_subnet" in x]        
    else:
        ensembles = [x for x in ensembles if "random_subnet" not in x]
    if eval_multiple_random:
        ensembles = [x for x in ensembles if "multiple_subnets" in x]        
    else:
        ensembles = [x for x in ensembles if "multiple_subnets" not in x]
    models = ensembles + individuals
    models = [x for x in models if getUser(x) in USERS_FOR_MODELS]
    
    #append control models and remove any duplicates
    models += ["model_random0_fold_{}_l2.pkl".format(i) for i in [0,1,2,3]]
    models += ["equally_weighted_ensemble_fold_{}_l2.pkl".format(fold_) for fold_ in [0,1,2,3]]    
    
    models = list(set(models))
    USERS_FOR_CSVS = ['UG1','UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'random0']

    ##get the csvs (i.e. annotation sets) to use 
    if val_type == "val_set":
        csvs = os.listdir("csvs/phase1/cross_validation/")
        csvs = [x for x in csvs if "val" in x]
        csvs = ["csvs/phase1/cross_validation/val_fold_INSERT_FOLD_thresholding_" + str(threshold) + ".csv" for threshold in range(1,6)]
        csvs += ["csvs/phase1/cross_validation/val_" + str(user) + "_fold_INSERT_FOLD.csv" for user in USERS_FOR_CSVS]
    if val_type == "test_set":
        csvs = os.listdir("csvs/phase1/test_set/")
        csvs = ["csvs/phase1/test_set/" + x for x in csvs]
    csvs = [x for x in csvs if ".csv" in x]

    ##sort the models and csvs for formatting the grid, not necessary but good for visualization purposes
    ##put ensemble models at the front for visualization, starting with agreed by n models 
    models.sort()
    models = [x for x in models if "thresholding" in x] + [x for x in models if "thresholding" not in x]
    ##put consensus n CSVs in front for visualization 
    csvs.sort()
    csvs = [x for x in csvs if "thresholding" in x] + [x for x in csvs if "thresholding" not in x] 
    ##instantiate the mapps to record the different results
    AUPRC_mapps = {"all": {}, 0: {}, 1: {}, 2: {}} ##mapp of class type to dictionary of key: (model, dataset) to value: auprc score 
    AUROC_mapps = {"all": {}, 0: {}, 1: {}, 2: {}}
    accuracy_mapps = {0: {}, 1: {}, 2: {}}
    print("models to use: ", models)
    print("csvs to use: ", csvs)
    ##evaluate each model against each csv
    for model_name in models:
        ##find cross validation fold of this model (specified in the name of the model)
        cross_val_fold = int(model_name[model_name.find("fold_") + 5: model_name.find("fold_") + 6])
        print("model_name: ", model_name, "cross val fold: ", cross_val_fold)
        ## do model instantiations here to account for different model types (ensembles)
        if "ensemble" in model_name:
            is_equally_weighted = "equally_weighted" in model_name 
            uses_random = "random_subnet" in model_name
            uses_multiple_random_subnets = "multiple_rand_subnets" in model_name
            unfreeze_amateurs = "unfrozen" in model_name
            use_amateur = True if "unfrozen" in model_name else False
            model = instantiateEnsembleModel(equally_weighted=is_equally_weighted,use_random_subnet=uses_random,use_amateur=use_amateur,cross_fold=cross_val_fold,use_multiple_random_subnets=uses_multiple_random_subnets, unfreeze_amateurs=unfreeze_amateurs).cuda()
        model = torch.load(models_dir + model_name).cuda()
        ##iterate over each csv, evaluating the model
        for csv in csvs:
            print(model_name, csv)
            if val_type == "val_set":
                csv = csv.replace("INSERT_FOLD", str(cross_val_fold))
            untransformed_dataset = MultilabelDataset(csv, DATA_DIR, threshold=.99,  transform=data_transforms['dev'])
            validation_generator = torch.utils.data.DataLoader(untransformed_dataset, batch_size=64, shuffle=True, num_workers=num_workers)
            running_val_loss = 0
            running_val_total = 0
            running_val_corrects = torch.zeros(3) #3 elements
            running_val_preds = torch.Tensor(0)         
            running_val_labels = torch.Tensor(0)
            with torch.set_grad_enabled(False):
                for data in validation_generator:
                    inputs, labels, raw_labels, names = data
                    running_val_labels = torch.cat([running_val_labels, labels])
                    inputs = Variable(inputs.cuda(), requires_grad=False)
                    labels = Variable(labels.cuda())
                    outputs = model(inputs) 
                    predictions = (torch.sigmoid(outputs)>0.5).type(torch.cuda.FloatTensor)
                    loss = criterion(outputs, labels)
                    running_val_total += len(labels)
                    running_val_loss += loss.item()
                    running_val_corrects += torch.sum(predictions==labels, 0).data.type(torch.FloatTensor)
                    preds = torch.sigmoid(outputs)
                    preds = preds.data.cpu()
                    running_val_preds = torch.cat([running_val_preds, preds])
                val_acc = running_val_corrects / running_val_total
                val_loss = running_val_loss / float(running_val_total)
                for class_type in ["all",0,1,2]:
                    if class_type == "all":
                        running_val_preds_specific_class = running_val_preds.numpy().ravel()
                        running_val_labels_specific_class = running_val_labels.numpy().ravel()
                    else: #only look at the column specified by class_type
                        running_val_preds_specific_class = running_val_preds.numpy()[:,class_type].ravel()
                        running_val_labels_specific_class = running_val_labels.numpy()[:,class_type].ravel()
                    precision, recall, _ = precision_recall_curve(running_val_labels_specific_class, running_val_preds_specific_class)
                    val_auprc = auc(recall, precision)
                    val_auroc = roc_auc_score(running_val_labels_specific_class,running_val_preds_specific_class)
                    if val_type == "val_set":
                        print("class: " + str(class_type) + ", validation auprc: {:.4f}, validation auroc: {:.4f}".format(val_auprc,val_auroc))
                    if val_type == "test_set":
                        print("class: " + str(class_type) + ", test auprc: {:.4f}, test auroc: {:.4f}".format(val_auprc,val_auroc))
                    AUPRC_mapps[class_type][(model_name, csv)] = val_auprc
                    AUROC_mapps[class_type][(model_name, csv)] = val_auroc
                    if class_type != "all":
                        accuracy_mapps[class_type][(model_name, csv)] = val_acc[class_type]
    ##condense maps and pickle
    for mapps in [AUPRC_mapps, AUROC_mapps, accuracy_mapps]:
        if mapps == AUPRC_mapps:
            map_name = "AUPRC map"
        elif mapps == AUROC_mapps:
            map_name = "AUROC map"
        elif mapps == accuracy_mapps:
            map_name = "accuracy map"
        else:
            print("no map found")
        print("map name: ", map_name)
        ##iterate over the 4 mapps (one for each class and one for all classes)
        for amyloid_type in mapps.keys():
            print("/////////////////////////////////////////////////////////////////////////")
            print("amyloid_type: ", amyloid_type)
            mapp = mapps[amyloid_type] ##mapp is the individual map of either AUPRC, AUROC, or accuracy
            ##average the different validation folds
            cross_val_mapp = {} #key: (model name stripped of cross validation decorators, csv stripped of decorators), value: average over cross val
            cross_val_stats_mapp = {} #to hold average and std
            for key in mapp.keys():
                mod, eval_csv = key
                mod = mod[0:mod.find("_fold")] + mod[mod.find("_fold") + 8:]
                if val_type == "val_set":
                    eval_csv = eval_csv[0:eval_csv.find("_fold")] + eval_csv[eval_csv.find("_fold") + 7: ]
                if (mod, eval_csv) not in cross_val_mapp.keys():
                    cross_val_mapp[(mod, eval_csv)] = [mapp[key]]
                else:
                    cross_val_mapp[(mod, eval_csv)].append(mapp[key])
            for key in cross_val_mapp.keys():
                cross_val_stats_mapp[key] = (np.mean(cross_val_mapp[key]), np.std(cross_val_mapp[key])) 
                cross_val_mapp[key] = np.mean(cross_val_mapp[key])
            mapp = cross_val_mapp
            pickle.dump(cross_val_stats_mapp , open("pickles/mapp_" + map_name + "val_type" + val_type + "_class_" + str(amyloid_type) + "_random_ensemble_" + str(eval_random_ensembles) + "_multiple_randoms_" + str(eval_multiple_random) + ".p", "wb")) 

def getInterraterAgreement(exclude_amateurs=False, phase="phase1", test_set_only=False):
    """
    Finds interrater agreement for either phase1 or phase2.
    EXLUDE_AMATEURS: whether or not we should include the novice annotators in our calculations
    PHASE: either "phase1" or "phase2"
    TEST_SET_ONLY: if True, then will only perform calculations off of the test set of phase1
    """
    if test_set_only:
        assert phase == "phase1"
    if exclude_amateurs:
        USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    else:
        USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    am_types = ["cored", "diffuse", "CAA"]
    ##instantiate dict mapping key: user, key: img_name, value: tuple of class labels
    user_img_dict = {user: {} for user in USERS} 
    for user in USERS:
        if phase == "phase1":
            if test_set_only:
                user_df = pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(user))
            else:
                user_df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        if phase == "phase2":
            user_df = pd.read_csv("phase2/phase2_annotations_{}.csv".format(user))
        for index, row in user_df.iterrows():
            if phase == "phase1":
                img_name = row["imagename"]
            if phase == "phase2":
                img_name = row["tilename"].replace("phase1/blobs/", "")
            if img_name not in user_img_dict[user].keys():
                user_img_dict[user][img_name] = [(row["cored"], row["diffuse"], row["CAA"])]
            else:
                user_img_dict[user][img_name].append((row["cored"], row["diffuse"], row["CAA"]))
    ##find what's in common among raters 
    if exclude_amateurs:
        shared_images = list(set.intersection(set(user_img_dict["NP1"].keys()), set(user_img_dict["NP2"].keys()), set(user_img_dict["NP3"].keys()), set(user_img_dict["NP4"].keys()), set(user_img_dict["NP5"].keys())))
    else:
        shared_images = list(set.intersection(set(user_img_dict["UG1"].keys()), set(user_img_dict["UG2"].keys()), set(user_img_dict["NP1"].keys()), set(user_img_dict["NP2"].keys()), set(user_img_dict["NP3"].keys()), set(user_img_dict["NP4"].keys()), set(user_img_dict["NP5"].keys())))
    user_class_dict = {user: {} for user in USERS} #key user, key class, value: list of labels (in the same sequence as shared_images)
    for user in USERS:
        labels_list = []
        for img in shared_images:
            labels_list.append(user_img_dict[user][img].pop())
        labels_list = np.array(labels_list)
        user_class_dict[user]["cored"] = list(labels_list[:,0])
        user_class_dict[user]["diffuse"] = list(labels_list[:,1])
        user_class_dict[user]["CAA"] = list(labels_list[:,2])
    kappa_dict = {} ##create kapp dictionary: key: (user1, user2) pair, key: class, value: kappa stat 
    accuracy_dict = {} #create accuracy dictionary: key: (user1, user2) pair, key: class, value: accuracy
    kappa_class_lists = {"cored": [], "diffuse": [], "CAA": []}
    combos = list(combinations(USERS, 2)) + [(user, user) for user in USERS]
    for combo in combos:
        kappa_dict[combo] = {am_type: {} for am_type in am_types}
        accuracy_dict[combo] = {am_type: {} for am_type in am_types}
        u1, u2 = combo[0], combo[1]
        for am_type in am_types:
            p_o = getAccuracy(user_class_dict[u1][am_type], user_class_dict[u2][am_type])
            p_e = getChanceAgreement(user_class_dict[u1][am_type], user_class_dict[u2][am_type])
            kappa = (p_o - p_e) / float(1 - p_e)
            kappa_dict[combo][am_type] = kappa
            accuracy_dict[combo][am_type] = p_o 
            if u1 != u2 and kappa != 1.0:
                kappa_class_lists[am_type].append(kappa)
    pickle.dump(kappa_dict ,open("pickles/{}_kappa_dict_exclude_novices_{}_test_set_only_{}.pk".format(phase, exclude_amateurs, test_set_only), "wb"))
    pickle.dump(accuracy_dict ,open("pickles/{}_accuracy_dict_exclude_novices_{}_test_set_only_{}.pk".format(phase, exclude_amateurs, test_set_only), "wb"))

def compareConsensusVsIndividual(mapp, eval_set="test_set", amyloid_type=0, comparison="truth", diagonal_only=False, exclude_consensus=False, exclude_amateurs=False, consensus_of_2_only=False, exclude_individuals=False):
    """
    Compares consensus model(s)/annotation sets with individual expert models/annotation sets, and pickles the results
    MAPP: the performance results mapp that is passed in (generated by the method grid_test)
    EVAL_SET: either "test_set" or "val_set" 
    AMYLOID_TYPE: either 0,1, or 2
    COMPARISON: either model or truth 
        if COMPARISON == "truth", will compare avg model performance over consensus annotation sets vs avg over individual annotation sets
        if COMPARISON == "model", will compare avg consenus model performance vs average individual model performance
    DIAGONAL_ONLY: if True, then will only look at entries that are a diagonal of a grid (i.e. model x, evaluated on annotation set x)
    EXCLUDE_CONSENSUS: if True, will not include consensus models if comparison == truth, and will not include consensus annotation sets if comparison == model
    EXCLUDE_AMATEURS: if True, will exclude the novice models and annotation sets 
    CONSENSUS_OF_2_ONLY: if True, will only keep the consensus of 2 model/annotation set as the consensus representative
    EXCLUDE_INDIVIDUALS: if True, will not include individual models if comparison == truth, and will not include individual annotation sets if comparison == model
    """
    consensus_entries = []
    consensus_stds = []
    individual_entries = []
    individual_stds = []
    if exclude_amateurs:
        mapp = {k:mapp[k] for k in mapp.keys() if "UG1" not in k[0] and "UG2" not in k[0]
            and "UG1" not in k[1] and "UG2" not in k[1]}
    if diagonal_only:
         mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[0]) == getUser(k[1])}
    if comparison == "model":
        index = 0
        if exclude_consensus:
            mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" not in k[1]}
        if exclude_individuals:
            mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[1]) not in ["UG1", "UG2", "NP1", "NP2", "NP3", "NP4", "NP5"]}
    if comparison == "truth":
        index = 1
        if exclude_consensus:
            mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" not in k[0]}
        if exclude_individuals:
            mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[0]) not in ["UG1", "UG2", "NP1", "NP2", "NP3", "NP4", "NP5"]}
    if consensus_of_2_only:
        mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" not in k[index] or ("thresholding_2" in k[index])}
    for key in mapp:
        if "thresholding" in key[index]:
            consensus_entries.append(mapp[key][0])
            consensus_stds.append(mapp[key][1])
        else:
            individual_entries.append(mapp[key][0])
            individual_stds.append(mapp[key][1])
    consensus_entries_avg, consensus_entries_std, consensus_entries_sample_size =  np.mean(consensus_entries), cochraneCombine(consensus_entries, consensus_stds), len(consensus_entries) * 4 ##each entry is an average of 4 cross val models applied to test set
    individual_entries_avg, individual_entries_std, individual_entries_sample_size =  np.mean(individual_entries), cochraneCombine(individual_entries, individual_stds), len(individual_entries) * 4
    am_types = {0: "cored", 1: "diffuse", 2:"CAA"} 
    pickle.dump((consensus_entries_avg, consensus_entries_std, consensus_entries_sample_size), open("pickles/comparison_consensus_vs_individual_consensus_stats_{}_{}_{}_exclude_consensus_{}_exclude_amateurs_{}_diagonal_only_{}_consensus_of_2_only_{}_exclude_individuals_{}.pk".format(comparison, eval_set, am_types[amyloid_type], exclude_consensus, exclude_amateurs, diagonal_only, consensus_of_2_only, exclude_individuals), "wb"))
    pickle.dump((individual_entries_avg, individual_entries_std, individual_entries_sample_size), open("pickles/comparison_consensus_vs_individual_individual_stats_{}_{}_{}_exclude_consensus_{}_exclude_amateurs_{}_diagonal_only_{}_consensus_of_2_only_{}_exclude_individuals_{}.pk".format(comparison, eval_set, am_types[amyloid_type], exclude_consensus, exclude_amateurs, diagonal_only, consensus_of_2_only, exclude_individuals), "wb"))

def generateAllCAMs(IMG_DIR=None, norm=None, save_dir="CAM_images/"):
    """
    Runs through the test set for each NP, novice, plus consensus of 2 model, and generates the CAM images for each class
    IMG_DIR: directory where images are located
    NORM: numpy object containing normalization data 
    SAVE_DIR: directory to save images to 
    """
    images_set = set()
    df_names = ["csvs/phase1/test_set/entire_test_thresholding_2.csv"] + ["csvs/phase1/cross_validation/train_duplicate_fold_{}_thresholding_2.csv".format(i) for i in range(0,4)]
    for df_name in df_names:
        df = pd.read_csv(df_name)
        df_images_set = set(df['imagename'])
        images_set.update(df_images_set)
    images_list = list(images_set)
    for np in ["UG1", "UG2"] + ["consensus_of_2"] + ["NP{}".format(i) for i in range(2,6)]:
        ##can use any fold model, if assess on test set 
        if np != "consensus_of_2":
            np_mod_name = "models/model_{}_fold_3_l2.pkl".format(np)
        else:
            np_mod_name = "model_all_fold_3_thresholding_2_l2.pkl"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not os.path.isdir(save_dir + "{}/".format(np)):
            os.mkdir(save_dir + "{}/".format(np))
        ##make 100 random directories in save_dir because writing so many files to a single directory will be very slow as directory gets large
        for i in range(0, 100):
            if not os.path.isdir(save_dir + "{}/{}/".format(np, i)):
                os.mkdir(save_dir + "{}/{}/".format(np, i))
        ##processing an image list this large is slow, let's break it up
        step = int(.01 * len(images_list))
        print("# of images: ", len(images_list))
        for i in range(0, 100):
            init_time = time.time()
            getGradCamImages(np_mod_name, image_list=images_list[i * step : (i+1) * step], save_images=True, target_classes=[0,1,2], norm=norm, IMG_DIR=IMG_DIR, save_dir=save_dir + "{}/".format(np))    
            print("time_elapsed for this round: ", time.time() - init_time)
        # ##now unpack all of the images in CAM_images/NP/ subdirectories to CAM_images/NP/ 
        for subdirectory in os.listdir(save_dir + "{}/".format(np)):
            for sub_file in os.listdir(save_dir + "{}/{}".format(np, subdirectory)):
                shutil.move(save_dir + "{}/{}/{}".format(np, subdirectory, sub_file), save_dir + "{}/{}".format(np, sub_file))
            ##remove temp 100 directories
            shutil.rmtree(save_dir + "{}/{}".format(np, subdirectory))

def getAmateurAndConsensusOf2Stats(amateur="UG1", truncated=False, image_dir="CAM_images/"):
    """
    Calculates relevant Tversky measures: tanimoto coefficients and dice, and also SSIM 
    Pickles the results
    AMATEUR: the novice to compare to 
    TRUNCATED: determines whether to run a truncated analysis over a smaller image set
    IMAGE_DIR: Where to find the grad CAM images 
    """
    amateur_mod_name = "models/model_{}_fold_3_l2.pkl".format(amateur)
    consensus2_mod_name = "models/model_all_fold_3_thresholding_2_l2.pkl"
    df = pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_2.csv")
    if truncated:
        images_list = list(df['imagename'])[0:5]
        random.shuffle(images_list)
    else:
        images_list = list(df['imagename'])
    granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
    larger_thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
    all_thresholds = sorted(list(set(granular_thresholds + larger_thresholds)))
    all_thresholds.remove(0) #thresh = 0 is meaningless 
    thresholds = larger_thresholds
    thresholds.remove(0)
    metrics = ["tanimoto", "dice", "SSIM"]
    thresh_dict = {am_class: {metric : {t: [] for t in all_thresholds} for metric in metrics} for am_class in [0,1,2]} #key metric: key: thresh, value: list of metrics
    total_pixels = float(256 * 256)   
    for amyloid_class in [0,1,2]:
        for t in all_thresholds:
            for image in images_list:
                image_name = image[image.find("/") + 1:image.find(".jpg")]
                try: 
                    consensus_CAM = np.array(Image.open(image_dir + "consensus_of_2/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, consensus2_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
                    amateur_CAM = np.array(Image.open(image_dir + amateur +  "/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, amateur_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
                except:
                    print("can't find: ")
                    print(image_dir + "{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, consensus2_mod_name.replace("models/", ""), amyloid_class))
                    print(image_dir + "{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, amateur_mod_name.replace("models/", ""), amyloid_class))
                    continue
                consensus_CAM[consensus_CAM >= t] = 255.0 
                consensus_CAM[consensus_CAM < t] = 0.0
                amateur_CAM[amateur_CAM >= t] = 255.0 
                amateur_CAM[amateur_CAM < t] = 0.0
                intersection = np.count_nonzero(consensus_CAM==amateur_CAM)# (consensus_CAM == amateur_CAM).sum()  
                tanimoto = intersection / (total_pixels + total_pixels  - intersection)
                dice = (2 * intersection) / (total_pixels + total_pixels)
                SSIM = ssim(consensus_CAM, amateur_CAM, data_range=amateur_CAM.max() - amateur_CAM.min()) #NaN valued at t = 0  
                if math.isnan(SSIM) or math.isnan(tanimoto) or math.isnan(dice):
                    print("nan detected, skipping example: {}".format(image_name))                    
                    print("threshold: {}, SSIM: {}, tanimoto: {}, dice: {}, intersection: {}, total pixels: {}".format(t, SSIM, tanimoto, dice, intersection, total_pixels))
                    print("novice CAM max: {}, novice CAM min: {}".format(amateur_CAM.max(), amateur_CAM.min()))
                    continue
                thresh_dict[amyloid_class]["tanimoto"][t].append(tanimoto)
                thresh_dict[amyloid_class]["dice"][t].append(dice)
                thresh_dict[amyloid_class]["SSIM"][t].append(SSIM)
    ##convert to dictionary of lists
    amyloid_to_metrics_dict = {am_type: {"t": [], "t_err":[], "d":[], "d_err":[], "s":[], "s_err":[]} for am_type in [0,1,2]} 
    for amyloid_class in [0,1,2]:
        amyloid_to_metrics_dict[amyloid_class]["t"] = [np.mean(thresh_dict[amyloid_class]["tanimoto"][t]) for t in thresholds]
        amyloid_to_metrics_dict[amyloid_class]["t_err"] = [np.std(thresh_dict[amyloid_class]["tanimoto"][t]) for t in thresholds]
        amyloid_to_metrics_dict[amyloid_class]["d"] = [np.mean(thresh_dict[amyloid_class]["dice"][t]) for t in thresholds]
        amyloid_to_metrics_dict[amyloid_class]["d_err"] = [np.std(thresh_dict[amyloid_class]["dice"][t]) for t in thresholds]
        amyloid_to_metrics_dict[amyloid_class]["s"] = [np.mean(thresh_dict[amyloid_class]["SSIM"][t]) for t in thresholds]
        amyloid_to_metrics_dict[amyloid_class]["s_err"]= [np.std(thresh_dict[amyloid_class]["SSIM"][t]) for t in thresholds]
    pickle.dump(amyloid_to_metrics_dict, open("pickles/CAM_amyloid_to_metrics_dict_{}.pk".format(amateur), "wb"))


def getAmateurWithConsensusOf2Difference(amateur="UG1", amyloid_class=0, truncated=False, image_dir="CAM_images/"):
    """
    Finds the difference between consensus of 2 CAM with the amateur CAM and pickles the results,
    Quantifies 3 classes: pixel on novice and off consensus "A", off novice and on consensus "C", on both or off both "B"  
    AMATEUR: the novice annotator
    AMYLOID_CLASS: the amyloid class to analyze 
    TRUNCATED: whether to run a shortened analysis over 50 images 
    IMAGE_DIR: where the CAM images are located
    """  
    amateur_mod_name = "models/model_{}_fold_3_l2.pkl".format(amateur)
    consensus2_mod_name = "models/model_all_fold_3_thresholding_2_l2.pkl"
    df = pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_2.csv")
    if truncated:
        images_list = list(df['imagename'])[0:5]
    else:
        images_list = list(df['imagename'])  
    ##iterate over the saved images in outputs and calculate dice coefficient for each consensus of 2 CAM, amateur CAM pair
    ##binarize the image with threshold t, we'll have two set of thresholds, one with a more granular search space, one without
    granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
    larger_thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
    all_thresholds = sorted(list(set(granular_thresholds + larger_thresholds)))
    thresh_dict = {t: {"A": -1, "B": -1, "C": -1} for t in all_thresholds} #key: thresh, key: result, must be one of these: (on novice and off consensus "A", off novice and on consensus "C", on both or off both "B"), value: tuple of (average %, std) 
    total_pixels = float(256 * 256)   
    for t in all_thresholds:
        on_amatuer_off_consensus, off_amatuer_on_consensus, same_both = [], [], []
        for image in images_list:
            image_name = image[image.find("/") + 1:image.find(".jpg")]
            try:
                consensus_CAM = np.array(Image.open(image_dir + "consensus_of_2/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, consensus2_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
                amateur_CAM = np.array(Image.open(image_dir + amateur +  "/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, amateur_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
            except:
                print("image CAMs not found for: {}".format(image_name))
                continue
            consensus_CAM[consensus_CAM >= t] = 255.0 
            consensus_CAM[consensus_CAM < t] = 0.0
            amateur_CAM[amateur_CAM >= t] = 255.0 
            amateur_CAM[amateur_CAM < t] = 0.0
            difference_CAM = np.subtract(consensus_CAM, amateur_CAM)
            on_amatuer_off_consensus.append(np.count_nonzero(difference_CAM == -255) / total_pixels)
            off_amatuer_on_consensus.append(np.count_nonzero(difference_CAM == 255) / total_pixels)
            same_both.append(np.count_nonzero(difference_CAM == 0) / total_pixels)
        thresh_dict[t]["A"] = (np.mean(on_amatuer_off_consensus), np.std(on_amatuer_off_consensus))
        thresh_dict[t]["B"] = (np.mean(same_both), np.std(same_both))
        thresh_dict[t]["C"] = (np.mean(off_amatuer_on_consensus), np.std(off_amatuer_on_consensus))
    pickle.dump(thresh_dict, open("pickles/CAM_threshold_stats_{}_{}.pkl".format(amateur, amyloid_class), "wb"))


def getSubsetPercentageConsensusOf2WithAmateur(amateur="UG1", amyloid_class=0, truncated=False, image_dir="outputs/"):
    """
     Will binarize the consensus of 2 CAM and the novice CAM at varying thresholds and compute the percentages, and pickle
        "consensus": positive pixels that are active in both consensus CAM and novice CAM / total # of positive pixels in consensus CAM
        "amateur": positive pixels that are active in both consensus CAM and novice CAM / total # of positive pixels in amateur CAM
    AMATEUR: the novice to compare to 
    AMYLOID_CLASS: the amyloid class to analyze
    TRUNCATED: determines whether to run a truncated analysis over a smaller image set
    IMAGE_DIR: Where to find the CAM images  
    """
    amateur_mod_name = "models/model_{}_fold_3_l2.pkl".format(amateur)
    consensus2_mod_name = "models/model_all_fold_3_thresholding_2_l2.pkl"
    df = pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_2.csv")
    if truncated:
        images_list = list(df['imagename'])[0:20]
    else:
        images_list = list(df['imagename'])
    granular_thresholds = list(np.arange(0,20, 2)) + list(np.arange(20, 260, 25)) + [255] #more granular search space
    larger_thresholds = list(np.arange(0,260, 15))  #uniform axis with more spaced out thresholds
    all_thresholds = sorted(list(set(granular_thresholds + larger_thresholds)))  
    thresh_dict = {mod: {t: [] for t in all_thresholds} for mod in ["amateur" , "consensus"]} #key: either amateur or consensus, key: threshold, value: list of percentages (that will be condensed to an average and std tuple after populating) 
    total_pixels = float(256 * 256)   
    for t in all_thresholds:
        for image in images_list:
            image_name = image[image.find("/") + 1:image.find(".jpg")]
            try:
                consensus_CAM = np.array(Image.open(image_dir + "consensus_of_2/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, consensus2_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
                amateur_CAM = np.array(Image.open(image_dir + amateur +  "/{}{}_class_{}_GGrad_Cam_gray_.jpg".format(image_name, amateur_mod_name.replace("models/", ""), amyloid_class)), dtype=np.int16)
            except:
                print("image CAMs not found for: {}".format(image_name))
                continue
            consensus_CAM[consensus_CAM >= t] = 255.0 
            consensus_CAM[consensus_CAM < t] = 0.0
            amateur_CAM[amateur_CAM >= t] = 255.0 
            amateur_CAM[amateur_CAM < t] = 0.0
            on_amatuer_on_consensus = np.count_nonzero((consensus_CAM == 255) & (amateur_CAM == 255))
            on_consensus = np.count_nonzero(consensus_CAM == 255)
            on_amateur = np.count_nonzero(amateur_CAM == 255)
            if on_consensus != 0 and on_amateur != 0:
                thresh_dict["consensus"][t].append(on_amatuer_on_consensus / float(on_consensus))
                thresh_dict["amateur"][t].append(on_amatuer_on_consensus / float(on_amateur))
        thresh_dict["consensus"][t] = (np.mean(thresh_dict["consensus"][t]), np.std(thresh_dict["consensus"][t]))
        thresh_dict["amateur"][t] = (np.mean(thresh_dict["amateur"][t]), np.std(thresh_dict["amateur"][t])) 
    pickle.dump(thresh_dict, open("pickles/CAM_subset_dict_{}_{}.pk".format(amateur, amyloid_class), "wb"))


def testEnsembleSuperiority(random_subnet=False, multiple_subnets=False):
    """
    Test if ensemble performance is on average better than non-ensembles
    Pickles performance differentials
    RANDOM_SUBNET: whether or not we're analyzing ensembles with a single random subnet included
    MULTIPLE_SUBNETS: whether or not we're analyzing ensembles with multiple random subnets included
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    consensus = [str(i) for i in range(1,6)]
    am_classes = [0,1,2]
    difference_map = {am_class: {mod_type : [] for mod_type in ["individual", "consensus"]} for am_class in am_classes} #key: amyloid class, key: individual or consensus, value: ensemble - singleton peformance difference list
    for am_class in am_classes:
        mapp = pickle.load(open("pickles/mapp_AUPRC mapval_typetest_set_class_{}_random_ensemble_{}_multiple_randoms_{}.p".format(am_class, random_subnet, multiple_subnets), "rb"))
        individuals = []
        ensembles = []
        for test in USERS + consensus: ##iterate over test sets
            for user in USERS + consensus: ##iterate over different models 
                ##fetch test set 
                if "NP" not in test:
                    test_csv = "csvs/phase1/test_set/entire_test_thresholding_{}.csv".format(test)
                else:
                    test_csv = "csvs/phase1/test_set/{}_test_set.csv".format(test)
                ##fetch single model and ensemble model 
                if "NP" not in user:
                    model = "model_allthresholding_{}_l2.pkl".format(user)
                    if random_subnet == False and multiple_subnets == False:
                        ensemble = "ensemble_model_allthresholding_{}_l2.pkl".format(user)
                    if multiple_subnets:
                        ensemble = "ensemble_use_multiple_subnets_model_allthresholding_{}_l2.pkl".format(user)
                    if random_subnet:
                        ensemble = "ensemble_random_subnet_model_allthresholding_{}_l2.pkl".format(user)
                else: 
                    model = "model_{}l2.pkl".format(user)
                    if random_subnet == False and multiple_subnets == False:
                        ensemble = "ensemble_model_{}l2.pkl".format(user)
                    if multiple_subnets:
                        ensemble = "ensemble_use_multiple_subnets_model_{}l2.pkl".format(user)
                    if random_subnet:
                        ensemble = "ensemble_random_subnet_model_{}l2.pkl".format(user)
                individual_perf = mapp[(model, test_csv)]
                ensembles_perf = mapp[(ensemble, test_csv)]
                individuals.append(individual_perf)
                ensembles.append(ensembles_perf)
                if "NP" not in test:
                    difference_map[am_class]["consensus"].append(ensembles_perf[0] - individual_perf[0])
                else:
                    difference_map[am_class]["individual"].append(ensembles_perf[0] - individual_perf[0])
    pickle.dump(difference_map, open("pickles/ensemble_superiority_difference_map_random_subnet_{}_multiple_subnets_{}.pkl".format(random_subnet, multiple_subnets), "wb"))


def compareGrids(score_type, amyloid_type, eval_set="test_set", exclude_amateurs=False, single_random=False, multiple_subnets=False):
    """
    Compares the normal ensemble results with either the ensemble model results with a single random subnet or with multiple random subnets, pickles the differences
    Pickles the performance differential results
    SCORE_TYPE: either "AUPRC" or "AUROC"
    AMYLOID_TYPE: either 0, 1, or 2
    EVAL_SET: either "test_set" or "eval_set"
    EXCLUDE_AMATEURS: whether we want to exclude the novice ensembles
    SINGLE_RANDOM: whether or not we're analyzing ensembles with a single random subnet included
    MULTIPLE_SUBNETS: whether or not we're analyzing ensembles with multiple random subnets included
    """
    assert(single_random != multiple_subnets)
    mapp_without_modification = pickle.load(open("pickles/mapp_AUPRC mapval_typetest_set_class_{}_random_ensemble_False_multiple_randoms_False.p".format(amyloid_type), "rb"))
    mapp_with_modification = pickle.load(open("pickles/mapp_AUPRC mapval_typetest_set_class_{}_random_ensemble_{}_multiple_randoms_{}.p".format(amyloid_type, single_random, multiple_subnets), "rb"))
    stds = []        
    for key in mapp_without_modification:
        stds.append(mapp_without_modification[key][1])
    ##filter the maps to get rid of random models and truth spaces, and also to only keep ensembles for comparison
    prohibited_models = ["model_random0l2.pkl", "equally_weighted_ensemblel2.pkl"]
    prohibited_truth_spaces = ["csvs/phase1/test_set/random_test_set.csv"]
    mapp_with_modification = {k:mapp_with_modification[k] for k in mapp_with_modification.keys() if k[0] not in prohibited_models and "ensemble" in k[0] and k[1] not in prohibited_truth_spaces}
    mapp_without_modification = {k:mapp_without_modification[k] for k in mapp_without_modification.keys() if k[0] not in prohibited_models and "ensemble" in k[0] and k[1] not in prohibited_truth_spaces}
    if exclude_amateurs:
        mapp_with_modification = {k:mapp_with_modification[k] for k in mapp_with_modification.keys() if getUser(k[0]) not in ["UG1", "UG2"] and getUser(k[1]) not in ["UG1", "UG2"]}
        mapp_without_modification = {k:mapp_without_modification[k] for k in mapp_without_modification.keys() if getUser(k[0]) not in ["UG1", "UG2"] and getUser(k[1]) not in ["UG1", "UG2"]}
    ##make sure we're comparing mapps with the same model, csv key set - need to adjust naming of models in mapp_with_modification to account for the fact that these models are appended with things like "random_subnet_"
    if single_random:
        set1 = set([(x[0].replace("random_subnet_", ""), x[1]) for x in mapp_with_modification.keys()])
        set2 = set([(x[0], x[1]) for x in mapp_without_modification.keys()])
        assert set1 == set2
    else:
        set1 = set([(x[0].replace("use_multiple_subnets_", ""), x[1]) for x in mapp_with_modification.keys()])
        set2 = set([(x[0], x[1]) for x in mapp_without_modification.keys()])
        assert set1 == set2
    difference_mapp = {}
    for key in mapp_with_modification.keys():
        model = key[0]
        if single_random:
            cleaned_model = model.replace("random_subnet_", "")
        else:
            cleaned_model = model.replace("use_multiple_subnets_", "")
        csv = key[1]
        difference_mapp[(cleaned_model, csv)] = mapp_with_modification[(model, csv)][0] - mapp_without_modification[(cleaned_model,csv)][0]
    difference_values = list(difference_mapp.values())
    difference_values = [abs(x) for x in difference_values]
    pickle.dump(difference_values, open("pickles/ensemble_difference_values_{}_random_{}_multiple_{}.pkl".format(amyloid_type, single_random, multiple_subnets), "wb"))


def getCountsOfNegativeFlagUnsure(phase="phase1"):
    """
    For each annotator, will get the counts of the times negative, flag, and not sure were selected for annotations from PHASE,
    including other statistics like how many of these times a positive annotation for an amyloid plaque was also selected,
    pickles results
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    categories = ["negative", "flag", "notsure", "negatives_with_pos", "flags_with_pos", "notsures_with_pos", "no_pos_marked"]
    subcategories = ["negative", "flag", "notsure"]
    subcategories_2 = ["negatives_with_pos", "flags_with_pos", "notsures_with_pos"]
    counts_dict = {u: { cat: 0 for cat in categories} for u in USERS}
    for user in USERS:
        if phase == "phase1":
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        else:
            df = pd.read_csv("csvs/phase2/annotations/phase2_comparison_{}.csv".format(user))
            del df['cored'] #these are model predictions or consensus counts from phase 1, don't want these
            del df['CAA']
            del df['diffuse']
            df = df.rename(columns={'negative annotation': 'negative', 'flag annotation': 'flag', 'notsure annotation':'notsure',
                'cored annotation': 'cored', 'diffuse annotation': 'diffuse', 'CAA annotation': 'CAA'})
        for item in subcategories:
            count = list(df[item]).count(1)
            counts_dict[user][item] += count
        for index, row in df.iterrows():
            marked_positive = row["cored"] == 1 or row["diffuse"] == 1 or row["CAA"] == 1
            if row["negative"] == 1 and marked_positive:
                counts_dict[user]["negatives_with_pos"] += 1
            if row["flag"] == 1 and marked_positive:
                counts_dict[user]["flags_with_pos"] += 1
            if row["notsure"] == 1 and marked_positive:
                counts_dict[user]["notsures_with_pos"] += 1
            if not marked_positive:
                counts_dict[user]["no_pos_marked"] += 1
    ##convert counts_dict into a dictionary of lists
    lists_dict = {k : [] for k in categories}
    for user in counts_dict:
        for category in categories:
            lists_dict[category].append(counts_dict[user][category])
    for k in lists_dict:
        lists_dict[k] = np.array(lists_dict[k])
    pickle.dump(lists_dict, open("pickles/negativeFlagNotSure_lists_dict_{}.pkl".format(phase), "wb"))
    pickle.dump(counts_dict, open("pickles/negativeFlagNotSure_counts_dict_{}.pkl".format(phase), "wb"))


def getAverageAgreementOfEachUser(exclude_amateurs=False, phase="phase1", test_set_only=False):
    """
    Get average kappa agreement for each user, pickles the results
    EXCLUDE_AMATEURS: whether to include the two undergraduates
    PHASE: either "phase1" or "phase2"
    TEST_SET_ONLY: whether we want stats on the test set only
    Requires "pickles/{PHASE}_kappa_dict_exclude_novices_{}_test_set_only_{}.pk"" from getInterraterAgreement
    """
    am_types = ["cored", "diffuse", "CAA"]
    if exclude_amateurs:
        USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    else:
        USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    kappa_dict = pickle.load(open("pickles/{}_kappa_dict_exclude_novices_{}_test_set_only_{}.pk".format(phase, exclude_amateurs, test_set_only), "rb"))
    if exclude_amateurs:
        kappa_dict = {key:kappa_dict[key] for key in kappa_dict.keys() if "UG1" not in [key[0], key[1]] and "UG2" not in [key[0], key[1]]}
    avg_dict = {am_type: [] for am_type in am_types} #key am_type, value: list of (user, avg score) tuples
    for am_type in am_types:
        for user in USERS:
            scores = []
            for u1, u2 in kappa_dict:
                if (user == u1 or user == u2) and kappa_dict[(u1,u2)][am_type] != 1:
                    scores.append(kappa_dict[(u1,u2)][am_type])
            avg_dict[am_type].append((user, float(str(np.mean(scores))[0:5])))
        sorted_avg_dict = sorted(avg_dict[am_type], key=lambda tup: tup[1])
        pickle.dump(sorted_avg_dict, open("pickles/{}_avg_sorted_user_kappa_{}_exclude_novices_{}_test_set_only_{}.pk".format(phase, am_type, exclude_amateurs, test_set_only), "wb"))


def getAverageSortedAxis(mapp, am_type, axis="row", exclude_self=False, exclude_consensus=False, exclude_amateurs=False):
    """
    Given a map of (y category, x category): (value, std) from method grid_test, will print the average of each AXIS in sorted order
    AXIS: either "row" or "column"
    EXCLUDE_SELF: whether to exclude the entries of model X evaluated on annotation set X
    EXCLUDE_CONSENSUS: whether to exclude consensus annotation spaces and models 
    EXCLUDE_AMATEURS: whether to exclude UG1 and UG2
    """
    if axis == "row":
        axis_index = 0
    if axis == "column":
        axis_index = 1
    if exclude_self:
        mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[0]) != getUser(k[1])}
    if exclude_consensus:
        mapp = {k: mapp[k] for k in mapp.keys() if "thresholding" not in k[0] and "thresholding" not in k[1]}
    if exclude_amateurs:
        mapp = {k: mapp[k] for k in mapp.keys() if getUser(k[0]) not in ["UG1", "UG2"] and getUser(k[1]) not in ["UG1", "UG2"]}
    mapp = {k: mapp[k] for k in mapp.keys() if "ensemble" not in k[0] and "ensemble" not in k[1] and "random" not in k[0] and "random" not in k[1]}
    new_dict = {} #key: model, value: list of AUPRCs across truth spaces
    for item in mapp:
        if item[axis_index] not in new_dict.keys():
            new_dict[item[axis_index]] = [mapp[item][0]]
        else:
            new_dict[item[axis_index]].append(mapp[item][0])
    for item in new_dict:
        new_dict[item] = np.mean(new_dict[item])
    sorted_axis = [(k, str(new_dict[k])[0:6]) for k in sorted(new_dict, key=new_dict.get, reverse=True)]
    am_types = {0: "cored", 1: "diffuse", 2:"CAA"}
    pickle.dump(sorted_axis, open("pickles/avg_sorted_user_{}_{}_exclude_self_{}_exclude_consensus_{}_exclude_amateurs_{}.pk".format(axis, am_types[am_type], exclude_self, exclude_consensus, exclude_amateurs), "wb"))


def getEnsembleWeights(eval_random_subnets=False, eval_amateurs=False, eval_multiple=False):
    """
    For each ensemble model, gets the weights of the last affine layer to determine rater contribution
    EVAL_RANDOM_SUBNETS: if we want to analyze the ensembles which contain a single random subnet
    EVAL_AMATEURS: if we want to analyze models trained on novice annotations
    EVAL_MULTIPLE: if we want to analyze the ensembles which contain multiple random subnets
    """
    ##filter models to evaluate 
    models_dir = "models/"
    models = os.listdir(models_dir)
    models = [x for x in models if "ensemble" in x]
    if eval_random_subnets:
        models = [x for x in models if "random_subnet" in x]
    else:
        models = [x for x in models if "random_subnet" not in x]
    if eval_amateurs == False:
        models = [x for x in models if "UG1" not in x and "UG2" not in x]
    if eval_multiple:
        models = [x for x in models if "multiple" in x] 
    else:
        models = [x for x in models if "multiple" not in x]
    ##don't need the equally weighted one for obvious reasons
    models = [x for x in models if "equally_weighted" not in x] 
    models = [x for x in models if ".pkl" in x]
   ##generate each x_coord of the grid
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    if eval_random_subnets:
        USERS.append('random1')
    elif eval_multiple:
        USERS += ['random1','random2','random3','random4','random5']
    if eval_amateurs == False:
        USERS = [u for u in USERS if u != "UG1" and u != "UG2"]
    classes = [0,1,2]
    x_coords = [user + "_" + str(c) for user in USERS for c in classes]
    ##iterate over models and get weights and document them in the mapp
    ##want grid of weights for each model (axis 1) and each each class for each user (axis 2)
    mapp = {} #key: (model name, user_class#) value: float percentage 
    for model_name in models:
        ##initial instantiation details and params not so important, because interested in the loaded final affine weights
        cross_val_fold = int(model_name[model_name.find("fold_") + 5: model_name.find("fold_") + 6])
        is_equally_weighted = "equally_weighted" in model_name 
        uses_random = "random_subnet" in model_name
        uses_multiple_random_subnets = "multiple_rand_subnets" in model_name
        unfreeze_amateurs = "unfrozen" in model_name
        use_amateur = True if "unfrozen" in model_name else False
        model = instantiateEnsembleModel(equally_weighted=is_equally_weighted,use_random_subnet=uses_random,use_amateur=use_amateur,cross_fold=cross_val_fold,use_multiple_random_subnets=uses_multiple_random_subnets, unfreeze_amateurs=unfreeze_amateurs).cuda()
        model = torch.load(models_dir + model_name).cuda()
        user_dict = getWeight(model, uses_random_subnet=eval_random_subnets,use_multiple=eval_multiple)
        for x in x_coords:
            u = x.split("_")[0]
            c = int(x.split("_")[1])
            mapp[(model_name, x)] = user_dict[u][c]
    ##if evaluating cross val, average the different validation folds
    cross_val_mapp = {} #key: (model name stripped of cross validation decorators, csv value) : average over cross val
    cross_val_stats_mapp = {} #to hold average and std (just to view)
    for key in mapp.keys():
        mod, x = key
        for fold in [0,1,2,3]:
            mod = mod.replace("fold_{}_".format(fold), "")
        if (mod, x) not in cross_val_mapp.keys():
            cross_val_mapp[(mod, x)] = [mapp[key]]
        else:
            cross_val_mapp[(mod, x)].append(mapp[key])
    for key in cross_val_mapp.keys():
        cross_val_stats_mapp[key] = (np.mean(cross_val_mapp[key]), np.std(cross_val_mapp[key])) 
        cross_val_mapp[key] = np.mean(cross_val_mapp[key])
    pickle.dump(cross_val_mapp , open("pickles/weights_random_subnet_" + str(eval_random_subnets) +  "_eval_amateurs_" + str(eval_amateurs) + "_multiple_" + str(eval_multiple) + ".p", "wb")) 

def getStainUserClassCounts(dataset="full"):
    """
    Saves dictionary that keeps track of counts of positive and negative annotations for each class type by stain
    DATASET either "full" or "test" (compute on either full annotation set, or just the test set)
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'thresholding_2']
    stain_types = ["UTSW", "UCD", "cw"]
    amyloid_map = {"cored":0, "diffuse":1, "CAA":2}
    #key: user, key: stain, key: class, value: (count of positive, count of negative annotations) 
    stain_user_class_counts_dict = {user: {stain_type: {amyloid_class: (-1, -1) for amyloid_class in [0,1,2]} for stain_type in stain_types} for user in USERS + ['thresholding_2']}
    test_images = list(pd.read_csv("csvs/phase1/test_set/entire_test_thresholding_2.csv")["imagename"])   
    for user in USERS:
        if user == "thresholding_2":
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_consensus_of_2.csv".format(user))
        else:
            df = pd.read_csv("csvs/phase1/binary_labels/phase1_labels_{}.csv".format(user))
        if dataset == "test":
            df = df[df['imagename'].isin(test_images)]
        for stain_type in stain_types:
            stain_df = df[df['imagename'].str.contains(stain_type)]
            for amyloid_class in ["cored", "diffuse", "CAA"]:
                class_annotations = stain_df[amyloid_class]
                pos_counts = len([x for x in class_annotations if float(x) >= 1])
                neg_counts = len([x for x in class_annotations if float(x) < 1])
                stain_user_class_counts_dict[user][stain_type][amyloid_map[amyloid_class]] = (pos_counts, neg_counts)
    pickle.dump(stain_user_class_counts_dict, open("pickles/stain_user_class_counts_dict_{}_set.pkl".format(dataset), "wb"))

def instantiateEnsembleModel(equally_weighted=False, use_random_subnet=False,use_amateur=False,cross_fold=-1,use_multiple_random_subnets=False, unfreeze_amateurs=False):
    """
    Instantiates and returns an ensemble model,
    If EQUALLY WEIGHTED is true, the ensemble model will be instantiated as an EquallyWeightedEnsembleNet, else model will be EnsembleNet 
    If USE_RANDOM_SUBNET, will also use a single constituent net trained on random labels, for a total of 6 ensembled single networks (or 8 if using the amateur nets)
    USE_AMATEUR indicates if the novice CNNs will be part of the network or not
    CROSS_FOLD indicates which cross validation fold we are using (used for pulling the correct constituent networks, ideally shouldn't mix models coming from different cross-validation folds)
    USE_MULTIPLE_SUBNETS indicates if we should use all 5 random subnets in the ensemble 
    UNFREEZE_AMATEURS will allow for training the novice CNN subnets 
    """
    base_indices = np.array([[1,0,0],[0,1,0],[0,0,1]]) #1 for keep index, 0 for mask out 
    if use_amateur and use_random_subnet:
        mask_shape = [8,1]
    if use_amateur and not use_random_subnet:
        mask_shape = [7,1]
    if not use_amateur and use_random_subnet:
        mask_shape = [6,1]
    if not use_amateur and not use_random_subnet:
        mask_shape = [5,1]
    if use_multiple_random_subnets and use_amateur:
        mask_shape = [12,1]
    if use_multiple_random_subnets and not use_amateur:
        mask_shape = [10,1]
    indices_mask = np.tile(base_indices,tuple(mask_shape)) #will be a stack of n = mask_shape[0] total repeats of base_indices, final shape: (n x 3) x 3
    model_dir = "models/"
    model1 = torch.load(model_dir + "model_UG1_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model2 = torch.load(model_dir + "model_NP5_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model3 = torch.load(model_dir + "model_UG2_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model4 = torch.load(model_dir + "model_NP4_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model5 = torch.load(model_dir + "model_NP2_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model6 = torch.load(model_dir + "model_NP3_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    model7 = torch.load(model_dir + "model_NP1_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')

    #models trained with older version of pytorch, need to adjust module for newer versions of pytorch
    for mod in [model1, model2, model3, model4,  model5,  model6,  model7]:
        for i, (name, module) in enumerate(mod._modules.items()):
            module = recursion_change_bn(mod)
    if use_random_subnet or use_multiple_random_subnets:
        model8 = torch.load(model_dir + "model_random0_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
    if use_multiple_random_subnets:
        model9 = torch.load(model_dir + "model_random1_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
        model10 = torch.load(model_dir + "model_random2_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
        model11 = torch.load(model_dir + "model_random3_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
        model12 = torch.load(model_dir + "model_random4_fold_{}_l2.pkl".format(cross_fold), map_location='cuda:0')
        if equally_weighted:
            model = EquallyWeightedEnsembleNet(model1, model2, model3, model4, model5, model6, model7,model8=model8,model9=model9,model10=model10,model11=model11,model12=model12,amateur_param=use_amateur)
        else: 
            model = EnsembleNet(indices_mask, model1, model2, model3, model4, model5, model6, model7,model8=model8,model9=model9,model10=model10,model11=model11,model12=model12, amateur_param=use_amateur)
    else:
        if equally_weighted:
            if use_random_subnet:
                model = EquallyWeightedEnsembleNet(model1, model2, model3, model4, model5, model6, model7,model8=model8,amateur_param=use_amateur)
            else:
                model = EquallyWeightedEnsembleNet(model1, model2, model3, model4, model5, model6, model7,amateur_param=use_amateur)
        else: 
            if use_random_subnet:
                model = EnsembleNet(indices_mask, model1, model2, model3, model4, model5, model6, model7,model8=model8, amateur_param=use_amateur)
            else:
                model = EnsembleNet(indices_mask, model1, model2, model3, model4, model5, model6, model7, amateur_param=use_amateur)
    ##freeze subnets 
    if unfreeze_amateurs == False:
        for param in model.model1.parameters():
            param.requires_grad = False
        for param in model.model3.parameters():
            param.requires_grad = False
    for param in model.model2.parameters():
        param.requires_grad = False
    for param in model.model4.parameters():
        param.requires_grad = False
    for param in model.model5.parameters():
        param.requires_grad = False
    for param in model.model6.parameters():
        param.requires_grad = False
    for param in model.model7.parameters():
        param.requires_grad = False
    if use_random_subnet or use_multiple_random_subnets:
        for param in model.model8.parameters():
            param.requires_grad = False
    if use_multiple_random_subnets:
        for param in model.model9.parameters():
            param.requires_grad = False
        for param in model.model10.parameters():
            param.requires_grad = False
        for param in model.model11.parameters():
            param.requires_grad = False
        for param in model.model12.parameters():
            param.requires_grad = False
    return model 

def train_model(model, criterion, optimizer, scheduler, user, fold, dataloaders=None, dataset_sizes=None, num_epochs=25, gpu_id=None):
    """
    Function to train the MODEL
    MODEL: the CNN, CRITERION: the loss function, OPTIMIZER: the optimization function
    SCHEDULER: a taining scheduler, USER: the annotator (or consensus), FOLD: fold of the cross validation,
    DATALOADERS: dictionary containing torch.utils.data.DataLoader with key: (user, fold) key:phase 
    DATASET_SIZES: dictionary containing dataset sizes with key: (user, fold) key:phase 
    NUM_EPOCHS: number of epochs to train for, GPU_ID: id of the GPU
    Returns the model that achieves the highest performance over the validation set 
    """
    since = time.time()
    best_loss = 10000.0
    best_cored_auprc = 0.0 #best validation cored ever obtained during training
    corresponding_CAA_auprc = 0.0 #the CAA AUPRC at the epoch in which the best cored AUPRC is obtained
    best_model = copy.deepcopy(model)
    for epoch in range(num_epochs):
        epoch_time = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  
            else:
                model.train(False)  
            running_loss = 0.0
            running_corrects = torch.zeros(3) #3 image classes
            running_preds = torch.Tensor(0)          
            running_labels = torch.Tensor(0)
            # Iterate over data.
            for data in dataloaders[user, fold][phase]:
                # get the inputs
                inputs, labels, raw_labels, names = data
                running_labels = torch.cat([running_labels, labels])
                # wrap them in Variable
                if torch.cuda.is_available():
                    if phase == 'train':
                        inputs = Variable(inputs.cuda(), requires_grad=True)
                    else:
                        inputs = Variable(inputs.cuda(), volatile=True)
                    labels = Variable(labels.cuda(), volatile=True)
                else:
                    if phase == 'train':
                        inputs =  Variable(inputs, requires_grad=True)
                    else:
                        inputs =  Variable(inputs, volatile=True)
                    labels = Variable(labels, volatile=True)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                preds = F.sigmoid(outputs) #posibility for each class
                if torch.cuda.is_available():
                    predictions = (F.sigmoid(outputs)>0.5).type(torch.cuda.FloatTensor)
                else:
                    predictions = (F.sigmoid(outputs)>0.5).type(torch.FloatTensor)
                loss = criterion(outputs, labels)
                preds = preds.data.cpu()
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.item()                
                running_corrects += torch.sum(predictions==labels, 0).data.type(torch.FloatTensor)
                running_preds = torch.cat([running_preds, preds])
            epoch_loss = running_loss / dataset_sizes[user, fold][phase]
            epoch_acc = running_corrects / dataset_sizes[user, fold][phase]
            AUPRCs = {0:-1, 1:-1, 2:-1}
            for amyloid_class in [0,1,2]:
                precision, recall, _ = precision_recall_curve(running_labels.numpy()[:,amyloid_class].ravel(), running_preds.numpy()[:,amyloid_class].ravel())
                auprc = auc(recall, precision)
                AUPRCs[amyloid_class] = auprc
            print('{} loss: {:.4f} AUPRCs: Cored: {:.4f} Diffuse: {:.4f} CAA: {:.4f}'.format(phase, epoch_loss, AUPRCs[0], AUPRCs[1], AUPRCs[2]))
            if phase == 'train':
                model.train_loss_curve.append(epoch_loss)
                model.train_auprc.append(auprc)
            else:
                model.dev_loss_curve.append(epoch_loss)
                model.dev_auprc.append(auprc)
                if best_cored_auprc < AUPRCs[0]:
                    best_model = copy.deepcopy(model)
                    best_cored_auprc = AUPRCs[0]
                    corresponding_CAA_auprc = AUPRCs[2]
        epoch_end = time.time() - epoch_time
        print('train, Epoch time {:.0f}m {:.0f}s'.format(
                epoch_end // 60, epoch_end % 60))
        print()
    print("best cored AUPRC: ", best_cored_auprc, ", corresponding CAA AUPRC: ", corresponding_CAA_auprc)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return best_model

def getWeight(model,uses_random_subnet=False,use_multiple=False):
    """
    Returns the models weights as a dictionary by user, key: user, value: list of weights
    If USES_RANDOM_SUBNET, then will also have weights of the random subnet used
    If USE_MULTIPLE, then will also have weights of the multiple random subnets used
    """
    if model.amateur_param == None or model.amateur_param == True: #if the models amatuer param is None, this comes from a model before this attribute was added, and when all ensembles had all 7 rater constituents (legacy artifact)
        if uses_random_subnet:
            USERS = ['UG1', 'NP5', 'UG2', 'NP4', 'NP2', 'NP3', 'NP1', 'random1']
        elif use_multiple:
            USERS = ['UG1', 'NP5', 'UG2', 'NP4', 'NP2', 'NP3', 'NP1', 'random1', 'random2', 'random3','random4','random5']
        else:
            USERS = ['UG1', 'NP5', 'UG2', 'NP4', 'NP2', 'NP3', 'NP1']
    if model.amateur_param != None and model.amateur_param == False:
        if uses_random_subnet:
            USERS = ['NP5', 'NP4', 'NP2', 'NP3', 'NP1', 'random1']
        elif use_multiple:
            USERS = ['NP5', 'NP4', 'NP2', 'NP3', 'NP1', 'random1', 'random2', 'random3','random4','random5']
        else:
            USERS = ['NP5', 'NP4', 'NP2', 'NP3', 'NP1']
    user_dict = {user:[] for user in USERS}
    if model.equally_weighted:
        return
    if model.ensemble:
        soft_weights = customSoftMax(model.weight)
        soft_weights = soft_weights.cpu().data.numpy()
        l0, l1, l2 = soft_weights[0], soft_weights[1], soft_weights[2] ##list of weights, one for each class
        l0, l1, l2 = l0[0::3], l1[1::3], l2[2::3] 
        for user in USERS:
            user_dict[user].append(l0[USERS.index(user)])
            user_dict[user].append(l1[USERS.index(user)])
            user_dict[user].append(l2[USERS.index(user)])
        avg_dict = {}
        for user in user_dict:
            mean = np.mean(user_dict[user])
            avg_dict[user] = mean
        return user_dict
    else:
        return -1 

def getGradCamImages(model_load_name, image_list=None, save_images=True, target_classes=[0,1,2], norm=None, IMG_DIR=None, save_dir="CAM_images/"):
    """
    Will calculate the CAM of each image and for each target class, and plot as greyscale images
    MODEL_LOAD_NAME: the model to load
    IMAGE_LIST: list of full path images to go over and derive saliency 
    SAVE_IMAGES: whether to save the images
    TARGET_CLASSES: list of the classes to derive saliency for, list containing [0,1, or 2] or any combination of these
    NORM: numpy object containing normalization data 
    IMG_DIR: the directory where the original images are located
    SAVE_DIR: directory to save images to 
    """
    short_mod_name = model_load_name[model_load_name.rfind("/") + 1:] #to be used for naming when saving
    if ".pt" in model_load_name:
        model = Net()
        checkpoint = torch.load(model_load_name, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model = torch.load(model_load_name)
    if image_list == None:
        file = pd.read_csv(CSV_DIR)
        image_list = list(file[(file['cored'] == 1) & (file['diffuse'] == 1)]['imagename'])[0:15]
    image_list = sorted(image_list)
    base_save_dir = save_dir
    for img in image_list:
        if img is np.nan:
            continue
        
        ##assign one of the 100 subdirectories to save img to  
        save_dir = base_save_dir + str(np.random.randint(0, 100)) + "/"
        
        wsi_name = img.split('/')[0]
        source_name = ''.join(img.split('/')[-1].split('.jpg'))
        img_name = wsi_name+'/'+source_name+'.jpg'
        original_image = cv2.imread(IMG_DIR+img_name, 1)
        cv2.imwrite(save_dir + source_name + "_og.jpg", original_image)
        im = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)    
        imtensor = transforms.ToTensor()(im)
        imtensor = transforms.Normalize(norm['mean'], norm['std'])(imtensor)
        imtensor = imtensor.view(1,3,256,256)
        input_img = Variable(imtensor, requires_grad=True)    
        
        for target_class in target_classes:
            # Guided Grad cam
            gcv2 = GradCam(model.cpu(), target_layer=23)
            cam = gcv2.generate_cam(input_img, target_class) #shape (256,256)
            GBP = GuidedBackprop(model.cpu())
            guided_grads = GBP.generate_gradients(input_img, target_class) #shape (3,256,256)
            cam_gb = guided_grad_cam(cam, guided_grads) #shape (3,256,256)
            grayscale_cam_gb = convert_to_grayscale(cam_gb) #shape (1,256,256)
            if save_images:
                save_gradient_images(grayscale_cam_gb, source_name + short_mod_name + '_class_' + str(target_class) + '_GGrad_Cam_gray_', save_dir)
    
def stratifyPerformanceByStainType(DATA_DIR=None, data_transforms=None, num_workers=None):
    """
    Calculates AUPRCs of each model on the test set, and stratifies performance by stain type, and amyloid class, pickles results 
    DATA_DIR: where raw images are located
    DATA_TRANSFORMS: dictionary of data transforms to apply, key: phase, value: transforms.compose object
    NUM_WORKERS: number of CPU cores to use  
    """
    USERS = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    pairs = [("csvs/phase1/test_set/entire_test_thresholding_2.csv","models/model_all_fold_3_thresholding_2_l2.pkl" )] + [("csvs/phase1/test_set/" + x + "_test_set.csv", "models/model_" + x + "_fold_3_l2.pkl") for x in USERS] 
    stain_types = ["UTSW", "UCD", "cw"]
    #key model name, key: stain type, key: amyloid class, value: AUPRC, AUROC    
    results = {getUser(model):{stain_type: {amyloid_class: 0 for amyloid_class in [0,1,2]} for stain_type in stain_types} for model in [pair[1] for pair in pairs]} 
    #key: stain type, value: count of instances 
    counts_dict = {stain_type: 0 for stain_type in stain_types} 
    for pair in pairs:
        csv = pair[0]
        modelName = pair[1]
        model = torch.load(modelName)
        color_normalized_dataset = MultilabelDataset(csv, DATA_DIR, threshold=.99,  transform=data_transforms['dev'])
        color_normalized_generator = torch.utils.data.DataLoader(color_normalized_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        validation_generator = color_normalized_generator
        ##run through test set and get predictions, and list of corresponding image names
        with torch.set_grad_enabled(False):
            running_val_total = 0
            running_val_corrects = torch.zeros(3) #3 elements
            running_val_preds = torch.Tensor(0)         
            running_val_labels = torch.Tensor(0)
            missed_examples = 0 
            img_names = []
            for data in validation_generator:
                inputs, labels, raw_labels, names = data
                running_val_labels = torch.cat([running_val_labels, labels])
                inputs = Variable(inputs.cuda(), requires_grad=False)
                labels = Variable(labels.cuda())
                outputs = model(inputs) 
                predictions = (torch.sigmoid(outputs)>0.5).type(torch.cuda.FloatTensor)
                running_val_total += len(labels)
                running_val_corrects += torch.sum(predictions==labels, 0).data.type(torch.FloatTensor)
                preds = torch.sigmoid(outputs)
                preds = preds.data.cpu()
                running_val_preds = torch.cat([running_val_preds, preds])
                img_names += list(names)
            ##now that we have our predictions, labels, and img_names, stratify results by amyloid class
            for class_type in [0,1,2]:   
                running_val_preds_specific_class = running_val_preds.numpy()[:,class_type].ravel()
                running_val_labels_specific_class = running_val_labels.numpy()[:,class_type].ravel()
                #stratify by stain type (institution)
                for stain_type in stain_types:
                    stain_preds, stain_labels = [], []
                    ##filter class specific preds/labels list to only include ones with designated stain type
                    for i in range(0, len(img_names)):
                        if stain_type in img_names[i]:
                            stain_preds.append(running_val_preds_specific_class[i])
                            stain_labels.append(running_val_labels_specific_class[i])
                    stain_preds, stain_labels = np.array(stain_preds), np.array(stain_labels)
                    precision, recall, _ = precision_recall_curve(stain_labels, stain_preds)
                    val_auprc = auc(recall, precision)
                    val_auroc = roc_auc_score(stain_labels, stain_preds)
                    results[getUser(modelName)][stain_type][class_type] = val_auprc, val_auroc
                    counts_dict[stain_type] = len(stain_preds)
    pickle.dump(results, open("pickles/compare_stain_types_images.pkl", "wb"))
    pickle.dump(counts_dict, open("pickles/stain_type_counts_dict_test_set.pkl", "wb"))

def getRandomAUPRCBaseline():
    """
    Calculates random AUPRC for individuals and consensus across test set
    Random AUPRC is P / (P + N)
    Saves this as a pickle of dictionary with key: "individual" or "consensus", key: amyloid_class, value: (mean, std)
    """
    amyloid_classes = ["cored", "diffuse", "CAA"]
    baseline_mapp = {typ : {amyloid_class: [] for amyloid_class in amyloid_classes } for typ in ["individual", "consensus"]}
    tests = ["NP{}".format(i) for i in range(1, 6)] + ["entire_test_thresholding_{}".format(i) for i in range(1,6)]
    for test in tests:
        if "NP" in test:
            df = pd.read_csv("csvs/phase1/test_set/{}_test_set.csv".format(test))
        if "thresholding" in test:
            df = pd.read_csv("csvs/phase1/test_set/{}.csv".format(test))
        for amyloid_class in amyloid_classes:
            positives = df[df[amyloid_class] >= 0.99][amyloid_class].count()
            negatives = len(df[amyloid_class]) - positives
            random = positives / float(positives + negatives)
            if "NP" in test:
                baseline_mapp["individual"][amyloid_class].append(random)
            if "thresholding" in test:
                baseline_mapp["consensus"][amyloid_class].append(random)
    ##average 
    for key in baseline_mapp:
        for amyloid_class in amyloid_classes:
            baseline_mapp[key][amyloid_class] = (np.mean(baseline_mapp[key][amyloid_class]), np.std(baseline_mapp[key][amyloid_class]))
    pickle.dump(baseline_mapp, open("pickles/random_AUPRC_baseline.pkl", "wb"))

##==================================================================
#3) HELPER FUNCTIONS
##==================================================================

def capitalizeEachWord(original_str):
    """
    ORIGINAL_STR of type string, will return each word of string with the first letter of each word capitalized
    """
    result = ""
    # Split the string and get all words in a list
    list_of_words = original_str.split()
    # Iterate over all elements in list
    for elem in list_of_words:
        # capitalize first letter of each word and add to a string
        if len(result) > 0:
            result = result + " " + elem.strip().capitalize()
        else:
            result = elem.capitalize()
    # If result is still empty then return original string else returned capitalized.
    if not result:
        return original_str
    else:
        return result

def recursion_change_bn(module):
    """
    Function to accomodate models trained with an old pytorch version, 
    for a given pytorch MODULE, sets the number of batches tracked parameter for the torch.nn.BatchNorm2d module
    """
    if isinstance(module, torch.nn.BatchNorm2d):
        module.num_batches_tracked = torch.Tensor(64)
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def getUser(string):
    """
    given a STRING, will find the user and return this user
    """
    USERS = ['NP1', 'NP2', 'NP3', 'NP4', 'NP5', 'UG1', 'UG2', 'random1', 'random2', 'random3', 'random4', 'random5']
    agreeds = ["thresholding_" + str(i) for i in range(1,6)]
    for u in USERS + agreeds:
        if u in string:
            return u
    return -1 

def customUserSort(l):
    """
    given list L, will return the same list, but sorted in the user order: 
    ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    And also keeping all ensembles behind non-ensembles
    Anything in the list that doesn't have a user keyword within it, will be added to the end of the list, in the same order that it occurs in the input list L
    """
    use_multiple_subnets = False
    for i in range(0, len(l)):
        if "multiple_subnets" in l[i]:
            use_multiple_subnets = True
    consensus_individs = ["thresholding_{}".format(i) for i in range(1, 6)]
    individs = ['UG1', 'UG2', 'NP1', 'NP2', 'NP3', 'NP4', 'NP5']
    consensus_ensembles = ["ensemble_model_all{}".format(ci) for ci in consensus_individs]
    individ_ensembles = ["ensemble_model_{}".format(i) for i in individs]
    USERS =   consensus_individs + individs + consensus_ensembles + individ_ensembles
    if use_multiple_subnets:
        consensus_mult_rand_ensembles = ["ensemble_use_multiple_subnets_model_all{}".format(ci) for ci in consensus_individs]
        individ_mult_rand_ensembles = ["ensemble_use_multiple_subnets_model_{}".format(i) for i in individs]
        USERS =  consensus_individs + individs + consensus_mult_rand_ensembles + individ_mult_rand_ensembles
    new_list = []
    indices_extracted = []
    for u in USERS:
        for i in range(0, len(l)):
            if u in l[i] and l[i] not in new_list:
                new_list.append(l[i])
                indices_extracted.append(i)
    indices_remaining = []
    for i in range(0, len(l)):
        if i not in indices_extracted:
            indices_remaining.append(i)
    for index in indices_remaining:
        new_list.append(l[index])
    assert len(new_list) == len(l)
    assert set(new_list) == set(l)
    return new_list

def customSoftMax(A, inplace=False):
    """
    Takes tensor A and applies softmax to tensor, except it excludes 0 valued floats and operates as if they are not there
    """
    if inplace:
        A[A==0] = float("-inf")
        A_softmax = F.softmax(A,dim=1)
        return A_softmax
    else:
        A_clone = A.clone().detach()
        A_clone[A_clone==0] = float("-inf")
        A_softmax = F.softmax(A_clone,dim=1)
        return A_softmax

def getAccuracy(l1, l2):
    """
    Returns accuracy as a percentage between two lists, L1 and L2, of the same length
    """
    assert(len(l1) == len(l2))
    return sum([1 for i in range(0, len(l1)) if l1[i] == l2[i]]) / float(len(l1))

def getFrequencyAccuracy(l1):
    """
    Returns accuracy as a percentage from a given list L1, such that accuracy is the frequency at which the majority item in the list occurs
    """
    count_zeros = sum([1 for i in range(0, len(l1)) if l1[i] == 0])
    count_ones = sum([1 for i in range(0, len(l1)) if l1[i] == 1])
    return max(count_zeros / float(len(l1)), count_ones / float(len(l1)))

def getChanceAgreement(l1, l2):
    """
    Returns p_e, the probability of chance agreement: (1/N^2) * sum(n_k1 * n_k2) for rater1, rater2, k categories (i.e. two in this case, 0 or 1), for two binary lists L1 and L2
    """
    assert(len(l1) == len(l2))
    summation = 0
    for label in [0, 1]:
        summation += l1.count(label) * l2.count(label)
    return (1 / float(len(l1)**2)) * summation

def getChanceAgreementFromList(l1):
    """
    Returns p_e, the probability of chance agreement: (1/N^2) * sum(n_k1 * n_k2) for k categories (i.e. two in this case, 0 or 1), from binary list L1
    """
    count_zeros = sum([1 for i in range(0, len(l1)) if l1[i] == 0])
    count_ones = sum([1 for i in range(0, len(l1)) if l1[i] == 1])
    return (1 / float(len(l1)**2)) * count_zeros * count_ones

def cochraneCombine(means, stds):
    """
    Given separate group MEANS and STDs, will combine them into one and return the std of the whole population
    N of pop assumed to be equal 4
    attribution: https://handbook-5-1.cochrane.org/chapter_7/table_7_7_a_formulae_for_combining_groups.htm
    """
    pop_mean = np.mean(means)
    first_sum = sum([3* std**2 for std in stds])
    second_sum = sum([4*(mean - pop_mean)**2 for mean in means])
    denom = (4 * len(means)) - 1
    population_std = np.sqrt((first_sum + second_sum) / float(denom))
    return population_std

def autolabel(rects, ax, percentage=False, fontsize=12):
    """
    Attach a text label above each bar in RECTS, displaying its height.
    AX: matplotlib axis
    PERCENTAGE: if labeling a bar graph with percentages
    FONTSIZE: fontsize to use
    """
    for rect in rects:
        height = rect.get_height()
        percentage_height = "{:,.0%}".format(height)
        if percentage:
            label = percentage_height
        else:
            label = height
        ax.annotate('{}'.format(label),
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontname="Times New Roman", fontsize=fontsize)

def convert_to_grayscale(cv2im):
    """
    Converts 3d image to grayscale
    Args:
        CV2IM (numpy arr): RGB image with shape (D,W,H)
    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    Attribution: https://github.com/utkuozbulak/pytorch-cnn-visualizations
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def save_gradient_images(gradient, file_name, save_dir):
    """
    Exports the original GRADIENT image
    GRADIENT: Numpy array of the gradient with shape (3, 224, 224)
    FILE_NAME: File name to be exported
    SAVE_DIR: Dir for saving
    """
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0) 
    cv2.imwrite(save_dir + file_name + ".jpg", gradient)

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
    Guided grad cam is just pointwise multiplication of cam mask and
    guided backprop mask, returns this multiplication
    GRAD_CAM_MASK: Class activation map mask
    GUIDED_BACKPROP_MASK:Guided backprop mask
    Attribution: https://github.com/utkuozbulak/pytorch-cnn-visualizations    
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def compareColorNormalizedTestingToUnNormalized(DATA_DIR=None, RAW_DATA_DIR=None, data_transforms=None, num_workers=None):
    """
    loads each model (fold 3), runs through the held-out test set and evaluates performance with color-normalized images,
    versus performance without color normalization, saves results in dictionary called "pickles/compare_color_norm_vs_unnorm.pkl"
    DATA_DIR: directory of color normalized images
    RAW_DATA_DIR: directory of unnormalized images
    DATA_TRANSFORMS dictionary specifying how to preprocess the image, key: phase (either "train" or "dev", value: transforms.Compose object
    NUM_WORKERS: How many cores to use for processing
    """
    USERS = ["UG{}".format(i) for i in [1,2]] + ["NP{}".format(i) for i in range(1,6)]
    pairs = [("csvs/phase1/test_set/entire_test_thresholding_2.csv","models/model_all_fold_3_thresholding_2_l2.pkl" )] + [("csvs/phase1/test_set/" + x + "_test_set.csv", "models/model_" + x + "_fold_3_l2.pkl") for x in USERS] 
    ##iterate over every pair of user csvs and models
    color_norm_dict = {is_norm : {u: {0: 0, 1:0, 2:0}  for u in USERS + ["thresholding_2"]} for is_norm in ["color_norm", "unnorm"]} #key: user, key: 1 of 3 classes, value: AUPRC over test set 
    for pair in pairs:
        csv = pair[0]
        modelName = pair[1]
        model = torch.load(modelName)
        color_normalized_dataset = MultilabelDataset(csv, DATA_DIR, threshold=.99,  transform=data_transforms['dev'])
        color_normalized_generator = torch.utils.data.DataLoader(color_normalized_dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        non_color_normalized_dataset = MultilabelDataset(csv, RAW_DATA_DIR, threshold=.99,  transform=data_transforms['dev'])
        non_color_normalized_generator = torch.utils.data.DataLoader(non_color_normalized_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

        with torch.set_grad_enabled(False):
            for validation_generator in [non_color_normalized_generator, color_normalized_generator]:
                running_val_total = 0
                running_val_corrects = torch.zeros(3) #3 elements
                running_val_preds = torch.Tensor(0)         
                running_val_labels = torch.Tensor(0)
                missed_examples = 0 
                for data in validation_generator:    
                    inputs, labels, raw_labels, names = data
                    running_val_labels = torch.cat([running_val_labels, labels])
                    inputs = Variable(inputs.cuda(), requires_grad=False)
                    labels = Variable(labels.cuda())
                    outputs = model(inputs) 
                    predictions = (torch.sigmoid(outputs)>0.5).type(torch.cuda.FloatTensor)
                    running_val_total += len(labels)
                    running_val_corrects += torch.sum(predictions==labels, 0).data.type(torch.FloatTensor)
                    preds = torch.sigmoid(outputs)
                    preds = preds.data.cpu()
                    running_val_preds = torch.cat([running_val_preds, preds])
                val_acc = running_val_corrects / running_val_total
                for class_type in [0,1,2]:   
                    running_val_preds_specific_class = running_val_preds.numpy()[:,class_type].ravel()
                    running_val_labels_specific_class = running_val_labels.numpy()[:,class_type].ravel()
                    precision, recall, _ = precision_recall_curve(running_val_labels_specific_class, running_val_preds_specific_class)
                    val_auprc = auc(recall, precision)
                    val_auroc = roc_auc_score(running_val_labels_specific_class,running_val_preds_specific_class)
                    if validation_generator == color_normalized_generator:
                        color_norm_dict["color_norm"][getUser(pair[0])][class_type] = val_auprc
                    if validation_generator == non_color_normalized_generator:
                        color_norm_dict["unnorm"][getUser(pair[0])][class_type] = val_auprc
    pickle.dump(color_norm_dict, open("pickles/compare_color_norm_vs_unnorm.pkl", "wb"))



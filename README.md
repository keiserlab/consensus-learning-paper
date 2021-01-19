# consensus-learning-paper

author: Daniel Wong (wongdanr@gmail.com)

## Open access image data
https://osf.io/xh2jd/ <br />
Identifier: 

## The following python packages are required: 
pyvips 2.1.2  <br />
libvips 8.2.2-1<br />
libgsf-dev>=1.14.27<br />
opencv 3.4.1<br />
pytorch 1.2.0<br />
torchvision 0.4.0<br />
numpy 1.16.4<br />
matplotlib 3.1.0 <br />
pandas 0.24.2<br />
PIL 6.1.0<br />
scikit-learn 0.21.2<br />
scikit-image 0.15.0<br />
scipy 1.3.0<br />
<br />
It is important to note that for preprocessing the WSIs, PyVips, libgsf-dev, and libvips must be the exactly the versions as specified, else results will differ slightly. For a detailed README on these packages and installing, please see pyvips_install_readme. 

## Hardware Requirements:
All deep learning models were trained using Nvidia Geforce GTX 1080 GPUs

## Code:

**preprocess_WSIs.py** Preprocesses the raw WSIs by color normalizing them and tiling them down to 1536 x 1535 pixel images.<br />

**blob_detect.py** Extract the 256 x 256 pixel, plaque-centered images from the 1536 x 1536 pixel images.<br />

**data_prep.py** Generates various CSVs and get things ready for the deep learning pipeline.<br /> 

**core.py** Contains the core class and method definitions for the whole study.<br />

**figure.py** Plots the figures shown in the paper.<br />

**train.py** Contains code for training both the single CNNs as well as the ensemble CNNs.<br />

**training_bash.sh** is a convenient bash script to train all of the models necessary for this study.<br />

**phase2.py** Contains code necessary for phase 2 of the study.<br />

**analysis.py** is the main runner code for the analysis of the entire study. <br />

**venn.py** is helper code to produce the venn diagram figure.<br />

**clear.py** clears figures/ and outputs/ directories.<br />

**normalize.py** performs the Reinhard color normalization process <br />

**vips_utls.py** contains image preprocessing helper code relevant to PyVips.

## Other content: 
**models:**<br />
This folder contains the fully trained models. "Thresholding" refers to a consensus-of-n model. "Random subnet" refers to ensemble models that contain a single random constituent CNN. "Multiple subnets" refers to ensembles that contain 5 total random constituent CNNs. <br />

**csvs/**<br />
This folder contains the CSVs for the study <br />
&nbsp;&nbsp;&nbsp;&nbsp;**phase1/**<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**binary_labels/** contains the binary annotation labels (i.e. yes or no label of a plaque box)<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**cross_validation/** contains the cross-validation fold datasets, using the floating point labels. The datasets are class-balanced.<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**floating_point_labels/** contains continuous floating point labels for the data (by taking into account all bounding boxes that an annotator labels).<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**test_set/** contains floating point labels for the held-out test set.<br /> 
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**image_details_phase1.csv** contains a subset of the output from blob_detect.py, specifying things like plaque coordinates, tile coordinates, etc.<br /> 
&nbsp;&nbsp;&nbsp;&nbsp;**phase2/**<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**annotations/** contains the binary annotation labels for phase2 (column headers: cored annotation, diffuse annotation, CAA annotation). The CSVs also contain the model predictions on images used for enrichment (column headers: cored, diffuse, CAA)<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**floating_point_labels/** contains the intermediate floating point labels used to generate the final labels.<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**final_labels/** contains the final floating point labels used for model evaluation.<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**image_details_phase2.csv** contains a subset of the output from blob_detect.py pertaining to phase 2 images. Specifies things like plaque coordinates, tile coordinates, etc.<br />
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**individualModelPredictionsForConsensusImageSet.csv** contains model predictions on the images enriched for by the consensus-of-two model.<br />

**CAM_images/** contains the guided Grad-CAM images. <br /> 

**figures/** is the output directory to save figures. <br /> 

**outputs/** is a temporary scratch directory. <br /> 

**pickles/** is the output directory to save pickle files containing results and intermediate data. <br /> 

**software_packages/** contains the relevant software packages used for WSI preprocessing. <br /> 

**tile_seg/** contains the 256 x 256 pixel images. These images are used for training and evaluation. <br /> 

**utils/** contains image normalization data. <br /> 




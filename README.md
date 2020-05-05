# DNN-based Human Body Part Segmentation Tool for Images of Natural Scenes
A trained Deep Neural Network (DNN) tool for automatic segmentation of human body parts in images of natural scenes. This tool was built to improve eye tracking data analysis. For details, please refer to the accompanying arxiv document (). If you use this tool in a publication, please cite the arxiv document.

# Setup
* Clone this repository : `git clone --recursive https://github.com/nih-fmrif/MLT_Body_Part_Segmentation.git`
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Create environment : `conda env create -f pytorch_conda_environment.yml`
* Download the trained DNN (https://doi.org/10.35092/yhjc.12245324)
* Extract the file `model_175_0.561689198017_0.894362765766.pth` and place it in `body_part_segmentation/code/pytorch/models`

# Segment Example
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Run `OMP_NUM_THREADS={CPU_NUM} CUDA_VISIBLE_DEVICES={GPU_NUM} python pred_folder.py --image_folder=examples/inputs/ --output=examples/outputs/  --model=models/model_175_0.561689198017_0.894362765766.pth --usegpu --image_prefix=.jpg`
* `body_part_segmentation/code/pytorch/examples/outputs/example-pred.png` will contain the visualization of the predicted label for each pixel in the example image
* `body_part_segmentation/code/pytorch/examples/outputs/example-var.png` will contain the visulaization of the model uncertatiny for each pixel in the example image
* `body_part_segmentation/code/pytorch/examples/outputs/example-pred.mat` will contain the the predicted label for each pixel in the example image, stored in the "prediction" variable
* `body_part_segmentation/code/pytorch/examples/outputs/example-var.mat` will contain the visulaization of the model uncertatiny for each pixel in the example image, stored in the "variance" variable



# Segment Images in a Folder
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Run `OMP_NUM_THREADS={CPU_NUM} CUDA_VISIBLE_DEVICES={GPU_NUM} python pred_folder.py --image_folder={IMAGE_FOLDER_PATH} --output={OUTPUT_FOLDER_PATH}  --model=models/model_175_0.561689198017_0.894362765766.pth --usegpu --image_prefix={.png or .jpg}`

# Training
* Download [Pascal-Part Annotations](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html) and [Pascal VOC 2010 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit) to "body_part_segmentation/data/raw" then extract tar files.
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Run `OMP_NUM_THREADS={CPU_NUM} CUDA_VISIBLE_DEVICES={GPU_NUM} python train.py - -usegpu`


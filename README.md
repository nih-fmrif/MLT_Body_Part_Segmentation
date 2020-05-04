# DNN-based Human Body Part Segmentation Tool for Images of Natural Scenes

For details, please refer to the accompanying arxiv docuemt (). If you use this tool in a publication, please cite the arxiv document.

# Setup

* Clone this repository : `git clone --recursive https://github.com/nih-fmrif/MLT_Body_Part_Segmentation.git`
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Create environment : `conda env create -f pytorch_conda_environment.yml`


# Training
* Download [Pascal-Part Annotations](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html) and [Pascal VOC 2010 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit) to "body_part_segmentation/data/raw" then extract tar files.
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Run `OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES={GPU_NUM} python train.py - -usegpu`

# Predicting
* Download the trained DNN ().
* Get the MODEL_PATH by fing the path to the trained DNN file (model_71_0.581312119961_0.88002923737.pth).
* Go to the "body_part_segmentation/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Run `OMP_NUM_THREADS={CPU_NUM} CUDA_VISIBLE_DEVICES={GPU_NUM} python pred_folder.py --image_folder={IMAGE_FOLDER_PATH} --output={OUTPUT_FOLDER_PATH}  --model={MODEL_PATH} --usegpu --image_prefix={.png or .jpg}`

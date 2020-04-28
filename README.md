# DNN-based Human Body Part Segmentation Tool for Images of Natural Scenes Written

For details, please refer to the accompanying arxiv docuemt ().

# Setup

* Clone this repository : `git clone --recursive https://github.com/nih-fmrif/MLT_Body_Part_Segmentation.git`
* Download [Pascal-Part Annotations](http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html) and [Pascal VOC 2010 Dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2010/index.html#devkit) to "body_part_segmentation/data/raw" then extract tar files.
* Go to the "reseg-pytorch/code/pytorch" : `cd body_part_segmentation/code/pytorch`
* Download and install [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://conda.io/miniconda.html)
* Create environment : `conda env create -f pytorch_conda_environment.yml`

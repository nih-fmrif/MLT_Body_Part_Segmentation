import os, sys
import glob
import argparse
import numpy as np
from scipy.io import savemat
from PIL import Image, ImageOps

parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', required=True, help='path of the image folder')
parser.add_argument('--model', required=True, help='path of the model')
parser.add_argument('--usegpu', action='store_true', help='enables cuda to predict on gpu')
parser.add_argument('--output', required=True, help='path of the output directory')
parser.add_argument('--image_prefix', required=True, help='image prefix for the files in image_folder')
opt = parser.parse_args()

image_folder = opt.image_folder
model_path = opt.model
output_path = opt.output
image_prefix = opt.image_prefix

try:
    os.makedirs(output_path)
except:
    pass

image_paths = glob.glob(os.path.join(image_folder,'*'+image_prefix))

model_dir = os.path.dirname(model_path)
sys.path.insert(0, model_dir)

from lib import Model, Prediction
from settings import ModelSettings

ms = ModelSettings()

model = Model(ms.LABELS, (1,3,256,256), load_model_path=model_path, usegpu=opt.usegpu)
prediction = Prediction(ms.IMAGE_SIZE_HEIGHT, ms.IMAGE_SIZE_WIDTH, ms.MEAN, ms.STD, model)

for image_path in image_paths:
    image, pred, var, v = prediction.predict(image_path,n_samples = 10)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    vis_pred_array = np.array(pred)
    vis_pred_array = vis_pred_array * (255.0/np.max(vis_pred_array))
    vis_pred = Image.fromarray(vis_pred_array.astype('uint8'))
    vis_pred = ImageOps.colorize(vis_pred,(0,0,255),(255,0,0))
    
    vis_var_array = np.array(var)
    vis_var_array = vis_var_array * (255.0/np.max(vis_var_array))
    vis_var = Image.fromarray(vis_var_array.astype('uint8'))
    vis_var = ImageOps.colorize(vis_var,(0,0,255),(255,0,0))
    
    #image.save(os.path.join(output_path, image_name + '.png'))
    pred.save(os.path.join(output_path, image_name + '-pred.png'))
    var.save(os.path.join(output_path, image_name + '-var.png'))
    #vis_pred.save(os.path.join(output_path, image_name + '-vis-pred.png'))
    #vis_var.save(os.path.join(output_path, image_name + '-vis-var.png'))
    #savemat(os.path.join(output_path, image_name + '-var.mat'),{'variance': v})
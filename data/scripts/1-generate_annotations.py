import os, glob
from PIL import Image
import numpy as np
from scipy.io import loadmat

DATA_DIR = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
IMG_DIR = os.path.join(DATA_DIR, 'raw', 'VOCdevkit', 'VOC2010', 'JPEGImages')
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed', 'annotations')

label_dict = {
    'hair': 1,
    'head': 2,
    'lear': 3,
    'leye': 4,
    'lebrow': 5,
    'lfoot': 6,
    'lhand': 7,
    'llarm': 7,
    'llleg': 6,
    'luarm': 7,
    'luleg': 6,
    'mouth': 8,
    'neck': 9,
    'nose': 10,
    'rear': 3,
    'reye': 4,
    'rebrow': 5,
    'rfoot': 6,
    'rhand': 7,
    'rlarm': 7,
    'rlleg': 6,
    'ruarm': 7,
    'ruleg': 6,
    'torso': 11
}

try:
    os.makedirs(OUTPUT_DIR)
except:
    pass

ann_files = glob.glob(os.path.join(DATA_DIR, 'raw', 'Annotations_Part', '*.mat'))

for ann_file in ann_files:
    img_name = os.path.splitext(os.path.basename(ann_file))[0]

    anns = loadmat(ann_file)['anno'][0][0][1][0]

    persons = [[ann[2], ann[3]] for ann in anns if ann[0][0] == 'person']

    if len(persons) == 0:
        continue

    img = Image.open(os.path.join(IMG_DIR, img_name + '.jpg'))
    img_width, img_height = img.size

    head_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for person in persons:
        #person[0] -> person mask
        #person[1] -> parts

        if len(person[1]) == 0:
            continue

        parts = person[1][0]
        for part in parts:
            part_name, part_ann = part
            part_name = part_name[0]
            
            head_mask[part_ann == 1] = label_dict[part_name]

    head_mask_img = Image.fromarray(head_mask)

    head_mask_img.save(os.path.join(OUTPUT_DIR, img_name + '.png'))

"""
    Show ground-truth of an image
"""

from CONFIG import Config

import os
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm


config = Config()

def visual_gt():
    i = random.randint(0, len(os.listdir(config.IMG_FOLDER))-1)
    test_image_path = os.path.join(config.IMG_FOLDER, os.listdir(config.IMG_FOLDER)[i])
    test_annot_path = os.path.join(config.ANNOT_FOLDER, os.listdir(config.ANNOT_FOLDER)[i])

    contents_xml = open(test_annot_path).read()
    soup = BeautifulSoup(contents_xml, 'html.parser')

    # get the size of image
    img_width = int(soup.find('width').string)
    img_height = int(soup.find('height').string)

    ground_truth_class_indices = []
    ground_truth_boxes = []

    # get class, bounding box of all objects in image
    for obj in soup.find_all('object'):

        # get ground-truth class
        class_name = obj.find('name').string
        class_index = config.CLASSES.index(class_name)
        ground_truth_class_indices.append(class_index)

        # get ground-truth box
        x_min = int(obj.find('xmin').string)
        x_max = int(obj.find('xmax').string)
        y_min = int(obj.find('ymin').string)
        y_max = int(obj.find('ymax').string)

        # truncate the part of bounding box that fall outsie the size of image
        x_min = max(0, x_min)
        x_max = min(img_width, x_max)
        y_min = max(0, y_min)
        y_max = min(img_height, y_max)

        ground_truth_boxes.append((x_min, y_min, x_max, y_max))

    
    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the region proposals
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.imshow(image)
    for i, (x_min, y_min, x_max, y_max) in enumerate(ground_truth_boxes):
        w = x_max - x_min
        h = y_max - y_min
        
        class_id = ground_truth_class_indices[i]
        class_name = config.CLASSES[class_id]

        color = config.COLORS[class_id]

        ax.add_patch(plt.Rectangle((x_min, y_min), w, h, edgecolor=color, alpha=0.5, lw=2, facecolor='none'))
        ax.text(x_min, y_min, f"{class_name}", c=color, fontdict={'fontsize':22})
    plt.tight_layout()
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    visual_gt()
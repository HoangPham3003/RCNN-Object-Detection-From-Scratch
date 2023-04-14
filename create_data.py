"""
    Step 1: Build object detection dataset with Selective Search
"""

from CONFIG import Config
from utils import compute_iou

import os
import random
import cv2
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm


config = Config()

image_name_list = os.listdir(config.IMG_FOLDER)
annot_name_list = os.listdir(config.ANNOT_FOLDER)

# Create data for training classification model

save_box_path = "./boxes_v2"
save_text_path = open("boxes_v2.txt", 'a')

total_boxes = 0

for i, annot_name in tqdm(enumerate(annot_name_list)):

    # get the full path of xml file
    full_annot_path = os.path.join(config.ANNOT_FOLDER, annot_name)
    full_img_path = os.path.join(config.IMG_FOLDER, image_name_list[i])

    # read the contents of xml file
    contents_xml = open(full_annot_path).read()
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

    # read image
    image = cv2.imread(full_img_path)

    # add gtbox as a positive image with a specified class (label)
    for j, gtbox in enumerate(ground_truth_boxes):

        class_index_roi = ground_truth_class_indices[j]
            
        gt_x_min, gt_y_min, gt_x_max, gt_y_max = gtbox
        gt_roi = image[gt_y_min:gt_y_max, gt_x_min:gt_x_max]

        gt_box_name = f"{total_boxes}.jpg"
        gt_box_path = os.path.join(save_box_path, gt_box_name)
        total_boxes += 1

        cv2.imwrite(gt_box_path, gt_roi)

        save_text_path.write(f"{class_index_roi} {gt_box_name}\n")

    # run selective search on the image and initialize the list of proposed boxes
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    proposedRects= []

    # loop for each box generated by Selective Search
    for (x_min, y_min, w, h) in rects:
        proposedRects.append((x_min, y_min, x_min + w, y_min + h))

    n_pos = 0
    n_neg = 0

    # loop proposed box
    for propRect in proposedRects[:min(len(proposedRects), config.MAX_PROPOSED_ROI)]:
        prop_x_min, prop_y_min, prop_x_max, prop_y_max = propRect

        area_propRect = (prop_x_max - prop_x_min + 1) * (prop_y_max - prop_y_min + 1)

        prop_box = None
        prop_box_name = None
        prop_box_path = None
        class_index = None

        chk_neg = False

        # loop for each ground-truth box
        for j, gtbox in enumerate(ground_truth_boxes):

            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gtbox

            class_index_roi = ground_truth_class_indices[j]

            iou = compute_iou(gtbox, propRect)

            # a proposed roi matches A ground-truth => positive
            if iou >= 0.6 and n_pos < config.LIMIT_POS_PROP_ROI:
                prop_box = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

                prop_box_name = f"{total_boxes}.jpg"
                prop_box_path = os.path.join(save_box_path, prop_box_name)
                
                class_index = class_index_roi
                break
        
            # check proposed roi inside totally in ground-truth
            fullOverlap = prop_x_min >= gt_x_min
            fullOverlap = fullOverlap and prop_y_min >= gt_y_min
            fullOverlap = fullOverlap and prop_x_max <= gt_x_max
            fullOverlap = fullOverlap and prop_y_max <= gt_y_max

            # check ground-truth inside totally in proposed roi
            fullOverlap_2 = gt_x_min >= prop_x_min
            fullOverlap_2 = fullOverlap_2 and gt_y_min >= prop_y_min
            fullOverlap_2 = fullOverlap_2 and gt_x_max <= prop_x_max
            fullOverlap_2 = fullOverlap_2 and gt_y_max <= prop_y_max

            # a proposed roi doesn't match ALL ground-truth => negative
            if not fullOverlap and not fullOverlap_2 and iou <= 0.05 and n_neg < config.LIMIT_NEG_PROP_ROI and area_propRect >= 1000:
                chk_neg = True
                
        if chk_neg:
            prop_box = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

            prop_box_name = f"{total_boxes}.jpg"
            prop_box_path = os.path.join(save_box_path, prop_box_name)
            
            class_index = 0
        
        if prop_box is not None and prop_box_name is not None and prop_box_path is not None and class_index is not None:
            total_boxes += 1
            cv2.imwrite(prop_box_path, prop_box)
            save_text_path.write(f"{class_index} {prop_box_name}\n")
            if class_index == 0:
                n_neg += 1
            else:
                n_pos += 1


save_text_path.close()

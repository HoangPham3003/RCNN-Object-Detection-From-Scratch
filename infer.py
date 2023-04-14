import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from tqdm import tqdm

from CONFIG import Config
from utils import compute_iou, overLapped

config = Config()


class InferDataset(Dataset):
    def __init__(self, prop_array=None,
                       infer_transform=None):
        self.prop_array = prop_array
        self.infer_transform = infer_transform

    def __len__(self):
        return len(self.prop_array)

    def __getitem__(self, index):
        prop_roi = self.prop_array[index]
        prop_roi = cv2.cvtColor(prop_roi, cv2.COLOR_BGR2RGB)
        pil_roi = Image.fromarray(prop_roi)
        if self.infer_transform:
            roi_transformed = self.infer_transform(pil_roi)
        return roi_transformed


# ==========================================================================
# Define infer model + infer transform
# ==========================================================================
infer_model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
infer_model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=512, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=512, out_features=128, bias=True),
    nn.ReLU(inplace=True),
    nn.Linear(in_features=128, out_features=config.NUM_CLASSES, bias=True),
)
infer_model.load_state_dict(torch.load(".best.pt"))
infer_model.to(config.DEVICE)
infer_model.eval()


mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

infer_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])


image_path =- "./images/fruit17.png"
    
image = cv2.imread(image_path)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
rects = ss.process()

proposals = []
rois = []

for propRect in rects[:min(len(rects), config.MAX_PROP_INFER_ROI)]:
    prop_x_min, prop_y_min, prop_w, prop_h = propRect
    prop_x_max = prop_x_min + prop_w
    prop_y_max = prop_y_min + prop_h

    roi = image[prop_y_min:prop_y_max, prop_x_min:prop_x_max]

    proposals.append(roi)
    rois.append((prop_x_min, prop_y_min, prop_x_max, prop_y_max))

proposals = np.asanyarray(proposals)
rois = np.asanyarray(rois)

infer_dataset = InferDataset(prop_array=proposals, infer_transform=infer_transform)

infer_loader = DataLoader(dataset=infer_dataset, batch_size=8, shuffle=False)

probabilities = []
for roi in tqdm(infer_loader):
    roi = roi.to(config.DEVICE)

    outputs = infer_model(roi)
    prob = F.softmax(outputs, dim=1)

    prob_np = prob.detach().cpu().numpy()

    for pr in prob_np:
        probabilities.append(pr)

probabilities = np.array(probabilities)
value_softmax, index_softmax = torch.max(torch.tensor(probabilities), dim=1)

index_softmax_arr = np.array(index_softmax)
value_softmax_arr = np.array(value_softmax)

# =====================================================================
# Non-Maximum Suppression (NMS) 
# =====================================================================

# Remove boxes that are predicted class 0 and value of softmax < 0.95
indices = np.where((index_softmax_arr != 0) & (value_softmax_arr > 0.95))[0]


value_softmax_detec = value_softmax_arr[indices]
index_softmax_detec = index_softmax_arr[indices]
prop_obj_detec = probabilities[indices]
rois_obj_detec = rois[indices]

unique_class = np.unique(index_softmax_detec)

# Remove redundant boxes
P = np.arange(0, len(rois_obj_detec))
keep = []

counter = 0
# loop for each ith class
while len(P) > 0:
    print(f"Loop {counter} : {len(P)}")
    counter += 1
    for idx_cl in unique_class:

        idx_sm_box_cl = np.where(index_softmax_detec == idx_cl)[0] # find all box that belong to class ith
        
        idx_sm_box_cl = np.intersect1d(idx_sm_box_cl, P)

        if len(idx_sm_box_cl) == 0:
            continue
        
        val_sm_box_cl = value_softmax_detec[idx_sm_box_cl]

        box_cl = rois_obj_detec[idx_sm_box_cl]

        sorted_val_sm_box_cl = np.argsort(val_sm_box_cl)
        
        max_val_sm_box_cl = val_sm_box_cl[sorted_val_sm_box_cl[-1]]

        max_box_cl = box_cl[sorted_val_sm_box_cl[-1]]

        max_idx_sm_box_cl = idx_sm_box_cl[sorted_val_sm_box_cl[-1]]


        P = np.delete(P, np.where(P == idx_sm_box_cl[sorted_val_sm_box_cl[-1]]))
        keep.append(idx_sm_box_cl[sorted_val_sm_box_cl[-1]])
        
        for i in P:
            if i in idx_sm_box_cl:
                b = rois_obj_detec[i]
                iou = compute_iou(max_box_cl, b)

                if iou >= config.NMS_THRESHOLD or overLapped(max_box_cl, b) == True:
                    P = np.delete(P, np.where(P == i))

           
index_softmax_detec_final = index_softmax_detec[keep]
value_softmax_detec_final = value_softmax_detec[keep]
prop_obj_detec_final = prop_obj_detec[keep]
rois_obj_detec_final = rois_obj_detec[keep]


image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the region proposals
fig, ax = plt.subplots(figsize=(15, 8))
ax.imshow(image)
for i, (x_min, y_min, x_max, y_max) in enumerate(rois_obj_detec_final):
    w = x_max - x_min
    h = y_max - y_min
    
    index_class = index_softmax_detec_final[i]
    class_name = config.CLASSES[index_class]
    color = config.COLOR[index_class]

    ax.add_patch(plt.Rectangle((x_min, y_min), w, h, edgecolor=color, alpha=0.5, lw=2, facecolor='none'))
    ax.text(x_min, y_min, f"{class_name}", c=color, fontdict={'fontsize':22})
plt.axis('off')
plt.show()

    


    

            

    
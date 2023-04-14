"""
    Compute IOU threshold of 2 boxes
"""

def compute_iou(box_a, box_b):
    """
      input:
        - box_a: [x_a_min, y_a_min, x_a_max, y_a_max]
        - box_b: [x_b_min, y_b_min, x_b_max, y_b_max]
    """
    x_a_min, y_a_min, x_a_max, y_a_max = box_a
    x_b_min, y_b_min, x_b_max, y_b_max = box_b

    # find (x, y) coordinates of intersection box
    x_inters_min = max(x_a_min, x_b_min)
    y_inters_min = max(y_a_min, y_b_min)
    x_inters_max = min(x_a_max, x_b_max)
    y_inters_max = min(y_a_max, y_b_max)

    # compute area of intersection
    inters_area = max(0, x_inters_max - x_inters_min + 1) * max(0, y_inters_max - y_inters_min + 1) # area = 0 in case 2 boxes don't have any intersection 

    # compute area of union = (box_a_area + box_b_area - inters_area)
    union_area = (x_a_max - x_a_min + 1) * (y_a_max - y_a_min + 1) +  (x_b_max - x_b_min + 1) * (y_b_max - y_b_min + 1) - inters_area

    # compute iou = inters_area / union_area
    iou = inters_area / union_area
    
    return iou


def overLapped(box_a, box_b):
    """
      input:
        - box_a: [x_a_min, y_a_min, x_a_max, y_a_max]
        - box_b: [x_b_min, y_b_min, x_b_max, y_b_max]
    """
    x_a_min, y_a_min, x_a_max, y_a_max = box_a
    x_b_min, y_b_min, x_b_max, y_b_max = box_b
    # check proposed roi inside totally in ground-truth
    fullOverlap = x_a_min >= x_b_min
    fullOverlap = fullOverlap and y_a_min >= y_b_min
    fullOverlap = fullOverlap and x_a_max <= x_b_max
    fullOverlap = fullOverlap and y_a_max <= y_b_max

    # check ground-truth inside totally in proposed roi
    fullOverlap_2 = x_b_min >= x_a_min
    fullOverlap_2 = fullOverlap_2 and y_b_min >= y_a_min
    fullOverlap_2 = fullOverlap_2 and x_b_max <= x_a_max
    fullOverlap_2 = fullOverlap_2 and y_b_max <= y_a_max

    if fullOverlap == False and not fullOverlap_2 == False:
        return False
    return True
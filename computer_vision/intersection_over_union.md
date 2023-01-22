# IoU (Intersection over Union)
- Reference: https://gaussian37.github.io/vision-detection-giou/
- To score how well the predicted box matches the ground-truth we can compute the IOU (or intersection-over-union, also known as the Jaccard index) between the two bounding boxes.
- Ideally, the predicted box and the ground-truth have an IOU of 100% but in practice anything over 50% is usually considered to be a correct prediction
- Normalizes the bounding box width and height by the image width and height so that they fall between 0 and 1.
- parametrizes the bounding box x and y coordinates to be offset of a particular grid cell location so they are also bounded between 0 and 1.
```python
def get_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin_inter = max(xmin1, xmin2)
    xmax_inter = min(xmax1, xmax2)
    ymin_inter = max(ymin1, ymin2)
    ymax_inter = min(ymax1, ymax2)
    
    area_inter = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    return iou


def get_giou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin_enclose, xmin_inter = sorted([xmin1, xmin2])
    ymin_enclose, ymin_inter = sorted([ymin1, ymin2])
    xmax_inter, xmax_enclose = sorted([xmax1, xmax2])
    ymax_inter, ymax_enclose = sorted([ymax1, ymax2])
    
    area_inter = max(0, xmax_inter - xmin_inter) * max(0, ymax_inter - ymin_inter)
    area_union = area1 + area2 - area_inter

    iou = area_inter / area_union
    
    area_enclose = (xmax_enclose - xmin_enclose) * (ymax_enclose - ymin_enclose)
    giou = iou - (area_enclose - area_union) / area_enclose
    return giou
```
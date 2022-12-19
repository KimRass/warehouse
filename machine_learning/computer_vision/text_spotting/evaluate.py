# Reference: https://gist.github.com/tarlen5/008809c3decf19313de216b9208f3734
import numpy as np


def get_giou(bbox1, bbox2):
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
    
    # iou = max(area_inter / area_union, np.finfo(np.float32).eps)
    iou = area_inter / area_union
    
    # xmin_enclose = min(xmin1, xmin2)
    # xmax_enclose = max(xmax1, xmax2)
    # ymin_enclose = min(ymin1, ymin2)
    # ymax_enclose = max(ymax1, ymax2)
    
    # area_enclose = max(0, (xmax_enclose - xmin_enclose) * (ymax_enclose - ymin_enclose))
    # return iou - (area_enclose - area_union) / area_enclose
    return iou


def get_f1_score(df_label, df_pred, iou_thr=0.5):
    gt_boxes = np.array(df_label[["xmin", "ymin", "xmax", "ymax"]])
    pred_boxes = np.array(df_pred[["xmin", "ymin", "xmax", "ymax"]])

    gt_idx_thr = list()
    pred_idx_thr = list()
    ious = list()
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = get_giou(pred_box, gt_box)
            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)
    else:
        gt_match_idx = []
        pred_match_idx = []
        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches
            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)
        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)
    # result = {'true_positive': tp, 'false_positive': fp, 'false_negative': fn}
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


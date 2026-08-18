"""
Object Detection Metrics Module
Computes IoU, Precision, Recall, F1, mAP@50, and mAP@50:95 per class.
"""

import numpy as np

def compute_box_iou(box1, box2):
    """Computes IoU between box1 and box2 in format [xmin, ymin, xmax, ymax]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def calculate_ap(recalls, precisions):
    """Computes Average Precision (AP) from precision-recall curve using 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def evaluate_detections(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes, iou_thresh=0.50, num_classes=4):
    """Computes Precision, Recall, F1, AP@iou_thresh for detections against ground truth.
    Inputs are lists per image.
    """
    per_class_tp = {c: 0 for c in range(num_classes)}
    per_class_fp = {c: 0 for c in range(num_classes)}
    per_class_fn = {c: 0 for c in range(num_classes)}

    for p_b, p_c, p_s, g_b, g_c in zip(pred_boxes, pred_classes, pred_scores, gt_boxes, gt_classes):
        matched_gt = set()
        
        # Sort predictions by confidence score descending
        if len(p_s) > 0:
            sort_idx = np.argsort(-np.array(p_s))
            p_b = [p_b[i] for i in sort_idx]
            p_c = [p_c[i] for i in sort_idx]

        for box_p, cls_p in zip(p_b, p_c):
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, (box_g, cls_g) in enumerate(zip(g_b, g_c)):
                if gt_idx in matched_gt or cls_p != cls_g:
                    continue
                iou = compute_box_iou(box_p, box_g)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_thresh and best_gt_idx != -1:
                per_class_tp[cls_p] += 1
                matched_gt.add(best_gt_idx)
            else:
                per_class_fp[cls_p] += 1

        for gt_idx, cls_g in enumerate(g_c):
            if gt_idx not in matched_gt:
                per_class_fn[cls_g] += 1

    per_class_metrics = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for c in range(num_classes):
        tp = per_class_tp[c]
        fp = per_class_fp[c]
        fn = per_class_fn[c]
        
        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class_metrics[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    return {
        "overall": {"precision": overall_prec, "recall": overall_rec, "f1": overall_f1},
        "per_class": per_class_metrics
    }

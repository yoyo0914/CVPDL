import json
import numpy as np
import argparse

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    iou = intersection / union if union > 0 else 0
    return iou

def calculate_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_threshold):
    pred_boxes = np.array(pred_boxes)
    gt_boxes = np.array(gt_boxes)
    
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    matched_gt = set()
    
    for i, pred_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if gt_labels[j] == pred_labels[i]:  # Only consider the same label
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # Debug print to show the IoU and the ground truth matching
        print(f'Prediction {i}: best IoU = {best_iou:.4f}, matched GT = {best_gt_idx}')
        
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            tp[i] = 1  # True positive
            matched_gt.add(best_gt_idx)
        else:
            fp[i] = 1  # False positive
    
    fn = len(gt_boxes) - len(matched_gt)  # False negatives
    
    # Debugging precision and recall counts
    print(f'True Positives: {sum(tp)}, False Positives: {sum(fp)}, False Negatives: {fn}')
    
    return tp, fp, fn

def calculate_ap(tp, fp, fn):
    if len(tp) == 0:
        precision = np.array([0])
        recall = np.array([0])
    else:
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Calculate precision and recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(float).eps)
        recall = tp_cumsum / (tp_cumsum + fn + np.finfo(float).eps)
    
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    method = "interp"  # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap

def calculate_map_per_instance(pred_data, gt_data, iou_thresholds=np.arange(0.5, 1.0, 0.05)):
    aps_per_instance = []
    aps_per_class = {}
    
    for instance in pred_data:
        pred_boxes = pred_data[instance]['boxes']
        pred_labels = pred_data[instance]['labels']
        
        gt_boxes = gt_data.get(instance, {}).get('boxes', [])
        gt_labels = gt_data.get(instance, {}).get('labels', [])
        
        if not gt_boxes and not pred_boxes:
            continue
        # Debug print to ensure predictions and ground truth are loaded correctly
        print(f'Instance {instance}: {len(pred_boxes)} pred boxes, {len(gt_boxes)} gt boxes')
        
        aps = []
        for iou_thresh in iou_thresholds:
            tp, fp, fn = calculate_precision_recall(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh)
            ap = calculate_ap(tp, fp, fn)
            aps.append(ap)
        
        aps_per_instance.append(np.mean(aps))
        
        # Calculate AP for each class
        unique_labels = set(pred_labels + gt_labels)
        for label in unique_labels:
            if label not in aps_per_class:
                aps_per_class[label] = []
            
            class_pred_boxes = [box for box, l in zip(pred_boxes, pred_labels) if l == label]
            class_pred_labels = [l for l in pred_labels if l == label]
            class_gt_boxes = [box for box, l in zip(gt_boxes, gt_labels) if l == label]
            class_gt_labels = [l for l in gt_labels if l == label]
            
            tp, fp, fn = calculate_precision_recall(class_pred_boxes, class_pred_labels, class_gt_boxes, class_gt_labels, 0.5)
            ap50 = calculate_ap(tp, fp, fn)
            
            tp, fp, fn = calculate_precision_recall(class_pred_boxes, class_pred_labels, class_gt_boxes, class_gt_labels, 0.75)
            ap75 = calculate_ap(tp, fp, fn)
            
            aps_per_class[label].append((ap50, ap75))
    
    return np.mean(aps_per_instance), aps_per_class

def calculate_class_ap(aps_per_class):
    class_ap = {}
    for label, aps in aps_per_class.items():
        ap50 = np.mean([ap[0] for ap in aps])
        ap75 = np.mean([ap[1] for ap in aps])
        class_ap[label] = {'AP50': ap50, 'AP75': ap75}
    return class_ap

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate mAP50-95 and class-wise AP50, AP75 for object detection.')
    parser.add_argument('pred_file', type=str, nargs='?', 
                    default=r'C:\Users\qazws\OneDrive\Desktop\eval\predictions_val.json', 
                    help='Prediction file path')
    parser.add_argument('gt_file', type=str, nargs='?', 
                    default=r'C:\Users\qazws\OneDrive\Desktop\eval\valid_target.json', 
                    help='Ground truth file path')
    
    args = parser.parse_args()

    pred_json = load_json(args.pred_file)
    gt_json = load_json(args.gt_file)

    mAP_50_95, aps_per_class = calculate_map_per_instance(pred_json, gt_json)
    class_ap = calculate_class_ap(aps_per_class)

    print("Class-wise AP50 and AP75:")
    print("{:<10} {:<10} {:<10}".format("Class", "AP50", "AP75"))
    print("-" * 30)
    
    # 排序
    sorted_classes = sorted(class_ap.keys(), key=lambda x: int(x))
    
    for label in sorted_classes:
        ap = class_ap[label]
        print("{:<10} {:<10.4f} {:<10.4f}".format(label, ap['AP50'], ap['AP75']))

    print(f'\nmAP(50-95): {mAP_50_95:.4f}')
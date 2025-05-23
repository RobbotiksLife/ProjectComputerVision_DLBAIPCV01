import torch
from torchvision.ops import box_iou
from tqdm import tqdm

import utils

def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0: # test for obvious results
        return [], list(range(len(gt_boxes))), list(range(len(pred_boxes)))

    iou_matrix = box_iou(pred_boxes, gt_boxes) # define the IOU metrix
    matched_gt = set()
    matched_pred = set()
    matches = []
    for pred_idx in range(iou_matrix.size(0)):
        ious = iou_matrix[pred_idx]
        max_iou, gt_idx = torch.max(ious, dim=0)

        if max_iou >= iou_threshold and gt_idx.item() not in matched_gt:
            matches.append((pred_idx, gt_idx.item()))
            matched_gt.add(gt_idx.item())
            matched_pred.add(pred_idx)
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]

    return matches, unmatched_gt, unmatched_pred

def combine_metrics(metrics):
    combined = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    for cls_metrics in metrics.values():
        for key in combined:
            combined[key] += cls_metrics[key]
    return combined

def evaluate_custom(model, dataset, get_transform_func, device='cuda', iou_threshold=0.5):
    num_classes = 3  # two classes + 1 something = 3
    metrics = {
        cls: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
        for cls in range(1, num_classes)
    }

    image_paths = [str(dataset.root + "/images/" + img_path) for img_path in dataset.imgs] # define image paths

    batch_size = 4 # can overload GPU so maybe change it if would create problems
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Evaluating"):
        batch_paths = image_paths[i:i+batch_size]
        batch_targets = [dataset[idx][1] for idx in range(i, i+len(batch_paths))]

        preds = utils.predict(batch_paths, model, get_transform_func, device=device)
        preds = utils.filter_predictions(preds, score_threshold=0.5)

        # define results
        for prediction, target in zip(preds, batch_targets):
            pred_boxes = prediction['boxes'].cpu()
            pred_labels = prediction['labels'].cpu()

            gt_boxes = target['boxes']
            gt_labels = target['labels']

            # define is boxes match
            matches, unmatched_gt, unmatched_pred = match_predictions(pred_boxes, gt_boxes, iou_threshold=iou_threshold)

            matched_gt_labels = set()
            matched_pred_labels = set()

            # if boxes match
            for pred_idx, gt_idx in matches:
                pred_label = pred_labels[pred_idx].item()
                gt_label = gt_labels[gt_idx].item()
                if pred_label == gt_label:
                    metrics[pred_label]['TP'] += 1
                else:
                    metrics[pred_label]['FP'] += 1
                    metrics[gt_label]['FN'] += 1
                matched_gt_labels.add(gt_label)
                matched_pred_labels.add(pred_label)

            # if boxes not match for predictions
            for idx in unmatched_pred:
                label = pred_labels[idx].item()
                metrics[label]['FP'] += 1

            # if boxes not match for labels
            for idx in unmatched_gt:
                label = gt_labels[idx].item()
                metrics[label]['FN'] += 1

            # Add TN - if class not in the image and it was not predicted than it is TN
            present_classes = set(gt_labels.tolist()) | set(pred_labels.tolist())
            for cls in range(1, num_classes):
                if cls not in present_classes:
                    metrics[cls]['TN'] += 1

    print("\n📊 Evaluation Results (per class):")
    metrics['all'] = combine_metrics(metrics)
    for cls, m in metrics.items():
        TP, FP, FN, TN = m['TP'], m['FP'], m['FN'], m['TN']
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (TP + TN) / (TP + FP + FN + TN + 1e-8)
        print(f"\nClass {cls}:")
        print(f"  TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")

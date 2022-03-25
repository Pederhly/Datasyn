import numpy as np
import matplotlib.pyplot as plt
from tools import read_predicted_boxes, read_ground_truth_boxes


def calculate_iou(prediction_box, gt_box):
    """Calculate intersection over union of single predicted and ground truth box.

    Args:
        prediction_box (np.array of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (np.array of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]

        returns:
            float: value of the intersection of union for the two boxes.
    """
    # YOUR CODE HERE
    x_max = min(prediction_box[2], gt_box[2])
    y_max = min(prediction_box[3], gt_box[3])
    x_min = max(prediction_box[0], gt_box[0])
    y_min = max(prediction_box[1], gt_box[1])

    # Compute intersection
    if(x_min > x_max or y_min > y_max):
        area_of_overlap = 0
    else:
        area_of_overlap = abs((x_max - x_min) * (y_max - y_min))

    # Compute union
    pb_a = abs((prediction_box[2] - prediction_box[0]) * (prediction_box[3] - prediction_box[1]))
    gt_a = abs((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))
    area_of_union = pb_a + gt_a - area_of_overlap

    # Compute iou
    iou = area_of_overlap/area_of_union
    assert iou >= 0 and iou <= 1
    
    return iou


def calculate_precision(num_tp, num_fp, num_fn):
    """ Calculates the precision for the given parameters.
        Returns 1 if num_tp + num_fp = 0

    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of precision
    """
    flag = num_tp + num_fp

    if flag == 0:
        return 1
    else:
        P = num_tp / (num_tp + num_fp)
        return P


def calculate_recall(num_tp, num_fp, num_fn):
    """ Calculates the recall for the given parameters.
        Returns 0 if num_tp + num_fn = 0
    Args:
        num_tp (float): number of true positives
        num_fp (float): number of false positives
        num_fn (float): number of false negatives
    Returns:
        float: value of recall
    """
    flag = num_tp + num_fn
    
    if flag == 0:
        return 0
    else:
        R = num_tp / (num_tp + num_fn)
        return R


class match: 
    def __init__(self, pred, gt, iou): 
        self.pred = pred
        self.gt = gt
        self.iou = iou

def get_all_box_matches(prediction_boxes, gt_boxes, iou_threshold):
    """Finds all possible matches for the predicted boxes to the ground truth boxes.
        No bounding box can have more than one match.

        Remember: Matching of bounding boxes should be done with decreasing IoU order!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns the matched boxes (in corresponding order):
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of box matches, 4].
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of box matches, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    """
    # Find all possible matches with a IoU >= iou threshold
    IOUs = np.array([])
    gt_indices = []
    pred_indices = []

    for pred in range(prediction_boxes.shape[0]):
        for gt in range(gt_boxes.shape[0]):
            IOU = calculate_iou(prediction_boxes[pred], gt_boxes[gt])
            if IOU>=iou_threshold:
                gt_indices.append(gt)
                pred_indices.append(pred)
                IOUs = np.append(IOUs, IOU)

    # Sort all matches on IoU in descending order
    decreasing_ious = np.flip(np.argsort(IOUs))
    
    # Find all matches with the highest IoU threshold
    pred_used = []
    gt_used = []
    pred_matched = np.empty((0,4),float)
    gt_matched =  np.empty((0,4),float)

    for i in decreasing_ious:  
        pred_index = pred_indices[i]
        gt_index = gt_indices[i]
        if (pred_index not in pred_used) and (gt_index not in gt_used):
            pred_used.append(pred_index)
            gt_used.append(gt_index)
            pred_matched = np.append(pred_matched, [prediction_boxes[pred_index]], axis = 0)
            gt_matched = np.append(gt_matched, [gt_boxes[gt_index]], axis = 0)
    
    return pred_matched, gt_matched


def calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes,
       calculates true positives, false positives and false negatives
       for a single image.
       NB: prediction_boxes and gt_boxes are not matched!

    Args:
        prediction_boxes: (np.array of floats): list of predicted bounding boxes
            shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        gt_boxes: (np.array of floats): list of bounding boxes ground truth
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        dict: containing true positives, false positives, true negatives, false negatives
            {"true_pos": int, "false_pos": int, false_neg": int}
    """

    count = {"true_pos": 0,
            "false_pos": 0,
            "false_neg": 0}

    match_prediction, match_gt = get_all_box_matches(prediction_boxes,gt_boxes,iou_threshold)
    
    count["true_pos"] = match_prediction.shape[0]
    count["false_pos"] = prediction_boxes.shape[0] - match_prediction.shape[0]
    count["false_neg"] = gt_boxes.shape[0] - match_gt.shape[0]

    return count


def calculate_precision_recall_all_images(
    all_prediction_boxes, all_gt_boxes, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates recall and precision over all images
       for a single image.
       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
    Returns:
        tuple: (precision, recall). Both float.
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0
    for prediction_boxes, gt_boxes in zip(all_prediction_boxes,all_gt_boxes):
        count = calculate_individual_image_result(prediction_boxes, gt_boxes, iou_threshold)
        true_pos += count["true_pos"]
        false_pos += count["false_pos"]
        false_neg += count["false_neg"]
    
    recall = calculate_recall(true_pos,false_pos,false_neg)
    precision = calculate_precision(true_pos,false_pos,false_neg)
    
    return precision, recall


def get_precision_recall_curve(
    all_prediction_boxes, all_gt_boxes, confidence_scores, iou_threshold):
    """Given a set of prediction boxes and ground truth boxes for all images,
       calculates the recall-precision curve over all images.
       for a single image.

       NB: all_prediction_boxes and all_gt_boxes are not matched!

    Args:
        all_prediction_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all predicted bounding boxes for the given image
            with shape: [number of predicted boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        all_gt_boxes: (list of np.array of floats): each element in the list
            is a np.array containing all ground truth bounding boxes for the given image
            objects with shape: [number of ground truth boxes, 4].
            Each row includes [xmin, ymin, xmax, ymax]
        scores: (list of np.array of floats): each element in the list
            is a np.array containting the confidence score for each of the
            predicted bounding box. Shape: [number of predicted boxes]

            E.g: score[0][1] is the confidence score for a predicted bounding box 1 in image 0.
    Returns:
        precisions, recalls: two np.ndarray with same shape.
    """
    # Instead of going over every possible confidence score threshold to compute the PR
    # curve, we will use an approximation
    confidence_thresholds = np.linspace(0, 1, 500)
    # YOUR CODE HERE
    precisions = [] 
    recalls = []

    for confidence_threshold in confidence_thresholds:
        all_boxes = []
        all_scores = []
        for box, score in zip(all_prediction_boxes, confidence_scores):
            approved_boxes = []
            approved_scores = []
            for i in range(box.shape[0]):
                if score[i] >= confidence_threshold:
                    approved_boxes.append(box[i])
                    approved_scores.append(score[i])
            all_boxes.append(np.array(approved_boxes))
            all_scores.append(np.array(approved_scores))
        
        all_prediction_boxes = all_boxes
        confidence_scores = all_scores

        all_precicions, all_recalls = calculate_precision_recall_all_images(all_boxes, all_gt_boxes, iou_threshold)
        precisions.append(all_precicions)
        recalls.append(all_recalls)
    
    return np.array(precisions), np.array(recalls)


def plot_precision_recall_curve(precisions, recalls):
    """Plots the precision recall curve.
        Save the figure to precision_recall_curve.png:
        'plt.savefig("precision_recall_curve.png")'

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        None
    """
    plt.figure(figsize=(20, 20))
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0.8, 1.0])
    plt.ylim([0.8, 1.0])
    plt.savefig("precision_recall_curve.png")


def calculate_mean_average_precision(precisions, recalls):
    """ Given a precision recall curve, calculates the mean average
        precision.

    Args:
        precisions: (np.array of floats) length of N
        recalls: (np.array of floats) length of N
    Returns:
        float: mean average precision
    """
    # Calculate the mean average precision given these recall levels.
    recall_levels = np.linspace(0.0, 1.0, 11)
    # Find it easier solving task with decresing value of recall
    recall_levels = np.flip(recall_levels)
    # YOUR CODE HERE
    prec = 0
    sum_prec = 0
    counter = 0

    for recall_lvl in recall_levels:
        while recalls[counter] > recall_lvl:
            if precisions[counter] > prec:
                prec = precisions[counter]
            counter += 1
        sum_prec += prec

    average_precision = sum_prec / 11
    return average_precision


def mean_average_precision(ground_truth_boxes, predicted_boxes):
    """ Calculates the mean average precision over the given dataset
        with IoU threshold of 0.5

    Args:
        ground_truth_boxes: (dict)
        {
            "img_id1": (np.array of float). Shape [number of GT boxes, 4]
        }
        predicted_boxes: (dict)
        {
            "img_id1": {
                "boxes": (np.array of float). Shape: [number of pred boxes, 4],
                "scores": (np.array of float). Shape: [number of pred boxes]
            }
        }
    """
    # DO NOT EDIT THIS CODE
    all_gt_boxes = []
    all_prediction_boxes = []
    confidence_scores = []

    for image_id in ground_truth_boxes.keys():
        pred_boxes = predicted_boxes[image_id]["boxes"]
        scores = predicted_boxes[image_id]["scores"]

        all_gt_boxes.append(ground_truth_boxes[image_id])
        all_prediction_boxes.append(pred_boxes)
        confidence_scores.append(scores)

    precisions, recalls = get_precision_recall_curve(
        all_prediction_boxes, all_gt_boxes, confidence_scores, 0.5)
    plot_precision_recall_curve(precisions, recalls)
    mean_average_precision = calculate_mean_average_precision(precisions, recalls)
    print("Mean average precision: {:.4f}".format(mean_average_precision))


if __name__ == "__main__":
    ground_truth_boxes = read_ground_truth_boxes()
    predicted_boxes = read_predicted_boxes()
    mean_average_precision(ground_truth_boxes, predicted_boxes)

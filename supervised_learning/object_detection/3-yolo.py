#!/usr/bin/env python3
"""Initialize Yolo"""
import numpy as np
import tensorflow.keras as K


class Yolo:
    """
    Class to perform object detection
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Constructor
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = f.read().strip().split('\n')
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Process output from darknet
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            box_confidences.append(
                K.activations.sigmoid(output[..., 4:5]).numpy())
            box_class_probs.append(
                K.activations.sigmoid(output[..., 5:]).numpy())
            grid_height, grid_width, anchor_boxes, _ = output.shape
            pw = self.anchors[i, :, 0][..., np.newaxis]
            ph = self.anchors[i, :, 1][..., np.newaxis]

            tw = np.exp(output[..., 2:3])
            th = np.exp(output[..., 3:4])

            b_w = np.concatenate((tw * pw) / self.model.input.shape[1], axis=0)
            b_h = np.concatenate((ph * th) / self.model.input.shape[2], axis=0)

            t_x = np.concatenate(
                K.activations.sigmoid(output[..., 0:1]).numpy(), axis=0)
            t_y = np.concatenate(
                K.activations.sigmoid(output[..., 1:2]).numpy(), axis=0)

            c_x = np.concatenate(
                np.repeat(
                    np.arange(0, grid_width)[np.newaxis, ...], grid_height,
                    axis=0))
            c_y = np.repeat(
                np.arange(0, grid_height)[np.newaxis, ...], grid_width,
                axis=1)[0]
            c_x = c_x[..., np.newaxis, np.newaxis]
            c_y = c_y[..., np.newaxis, np.newaxis]

            b_x = (((t_x + c_x) / grid_width) - (b_w / 2))
            b_y = (((t_y + c_y) / grid_height) - (b_h / 2))
            b_w += b_x
            b_h += b_y

            boxes.append(
                np.concatenate(
                    [b_x * image_size[1],
                     b_y * image_size[0],
                     b_w * image_size[1],
                     b_h * image_size[0]], axis=-1)
                .reshape(grid_height, grid_width, anchor_boxes, 4))
        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Prediction filtering
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            b_scores = box_confidences[i] * box_class_probs[i]
            b_classes = np.argmax(b_scores, axis=-1)
            b_score_max = np.max(b_scores, axis=-1)

            pobj = np.where(b_score_max >= self.class_t)

            filtered_boxes.append(boxes[i][pobj])
            box_classes.append(b_classes[pobj])
            box_scores.append(b_score_max[pobj])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Compute non-max suppression
        """
        indices = np.lexsort((-box_scores, box_classes))
        filtered_boxes = filtered_boxes[indices]
        box_scores = box_scores[indices]
        box_classes = box_classes[indices]

        unique_classes = np.unique(box_classes)
        nms_boxes = []
        nms_classes = []
        nms_scores = []

        for cls in unique_classes:
            cls_indices = np.where(box_classes == cls)[0]
            cls_boxes = filtered_boxes[cls_indices]
            cls_scores = box_scores[cls_indices]

            while len(cls_boxes) > 0:
                max_score_index = np.argmax(cls_scores)
                nms_boxes.append(cls_boxes[max_score_index])
                nms_classes.append(cls)
                nms_scores.append(cls_scores[max_score_index])

                if len(cls_boxes) == 1:
                    break

                cls_boxes = np.delete(cls_boxes, max_score_index, axis=0)
                cls_scores = np.delete(cls_scores, max_score_index)

                ious = self.IoU(nms_boxes[-1], cls_boxes)
                iou_indices = np.where(ious <= self.nms_t)[0]

                cls_boxes = cls_boxes[iou_indices]
                cls_scores = cls_scores[iou_indices]

        return (np.array(nms_boxes),
                np.array(nms_classes),
                np.array(nms_scores))

    def IoU(self, box1, box2):
        """
        Compute intersection over union
        """
        x1 = np.maximum(box1[0], box2[:, 0])
        y1 = np.maximum(box1[1], box2[:, 1])
        x2 = np.minimum(box1[2], box2[:, 2])
        y2 = np.minimum(box1[3], box2[:, 3])

        inter_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area

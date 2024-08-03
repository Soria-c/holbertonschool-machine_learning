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

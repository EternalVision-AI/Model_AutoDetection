import cv2
import os

import time
import numpy as np
import math


class YoloV3(object):
    def __int__(self, weights, cfg, size, classes, confidence=0.5, nms=0.4):
        self.weight = weights
        self.cfg = cfg
        self.size = size

        self.net = cv2.dnn.readNet(self.weight, self.cfg)
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            print(Exception)
        self.classes = classes
        self.conf_thresh = confidence
        self.nms_thresh = nms

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def inference(self, frame):

        fh, fw, fc = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, self.size, (0, 0, 0), True, crop=False)

        # set input blob for the network
        self.net.setInput(blob)

        # run inference through the network
        # and gather predictions from output layers
        outs = self.net.forward(self.get_output_layers())

        # initialization
        class_ids = []
        confidences = []
        boxes = []

        # for each detetion from each output layer
        # get the confidence, class id, bounding box params
        # and ignore weak detections (confidence < 0.5)
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_thresh:  # original value : 0.5
                    if math.isinf(detection[0]) or math.isinf(detection[1]) or math.isinf(detection[2]) or math.isinf(
                            detection[3]):
                        continue
                    if detection[0] is None or detection[1] is None or detection[2] is None or detection[3] is None:
                        continue

                    center_x = int(detection[0] * fw)
                    center_y = int(detection[1] * fh)
                    w = int(detection[2] * fw)
                    h = int(detection[3] * fh)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    if x < 2 or y < 2:
                        continue
                    if x + w + 2 > fw or h + y + 2 > fh:
                        continue

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # apply non-max suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)

        H, W = frame.shape[:2]
        detections = []
        for i in indices:
            index = i[0]
            box = boxes[index]
            if box[2] * box[3] / (H * W) > 0.00001:
                classid = class_ids[index]
                label = self.classes[classid]
                detection = {
                    'image_sz': (W, H),
                    'class_id': classid,
                    'class_name': label,
                    'confidence': confidences[index],
                    'box': box,
                    'scale': 1}
                detections.append(detection)
        return detections


class YoloV8:
    def __init__(self, onnx_path, msize, classes, conf=0.3, nms=0.4):
        self.Net = cv2.dnn.readNetFromONNX(onnx_path)
        try:
            self.Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            self.Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.msize = msize
        self.conf = conf
        self.nms = nms
        self.classes = classes

    def inference(self, input_image):

        [height, width, _] = input_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = input_image
        scale = length / (self.msize[0])

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, size=self.msize, swapRB=True)
        self.Net.setInput(blob)
        outputs = self.Net.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= self.conf:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.nms, 0.5)

        detections = []
        H, W = input_image.shape[:2]

        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            classid = class_ids[index]
            label = self.classes[class_ids[index]]
            detection = {
                'image_sz' : (W, H),
                'class_id': classid,
                'class_name': label,
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)

        return detections


class YoloV5:
    def __init__(self, onnx_path, msize, classes, conf, nms):

        self.Net = cv2.dnn.readNetFromONNX(onnx_path)
        try:
            self.Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            self.Net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
            self.Net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.msize = msize
        self.conf_thresh = conf
        self.nms_thresh = nms
        self.classes = classes

    def inference(self, input_image):

        blob = cv2.dnn.blobFromImage(input_image, scalefactor=1 / 255.0, size=self.msize, mean=[0, 0, 0], swapRB=True, crop=False)

        # Sets the input to the network.
        self.Net.setInput(blob)

        # Run the forward pass to get output of the output layers.
        outputs = self.Net.forward(self.Net.getUnconnectedOutLayersNames())

        # Lists to hold respective values while unwrapping.
        class_ids = []
        confidences = []
        boxes = []
        # Rows.
        rows = outputs[0].shape[1]
        image_height, image_width = input_image.shape[:2]
        # Resizing factor.
        x_factor = image_width / self.msize[0]
        y_factor = image_height / self.msize[1]
        # Iterate through detections.
        for r in range(rows):
            row = outputs[0][0][r]
            confidence = row[4]
            # Discard bad detections and continue.
            if confidence >= self.conf_thresh:
                classes_scores = row[5:]
                # Get the index of max class score.
                class_id = np.argmax(classes_scores)
                #  Continue if the class score is above threshold.
                if (classes_scores[class_id] > self.conf_thresh):
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    cx, cy, w, h = row[0], row[1], row[2], row[3]
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)
                    box = np.array([left, top, left + width, top + height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_thresh, self.nms_thresh)

        detections = []
        H, W = input_image.shape[:2]

        for i in indices:
            index = i
            box = boxes[index]
            classid = class_ids[index]
            label = self.classes[classid]
            detection = {
                'image_sz': (W, H),
                'class_id': classid,
                'class_name': label,
                'confidence': confidences[index],
                'box': box,
                'scale': 1}
            detections.append(detection)

        return detections






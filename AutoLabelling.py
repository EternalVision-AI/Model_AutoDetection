
import cv2
import os
import numpy as np
import ctypes  # An included library with Python install.


import time
from labellingEngine import makeLabelling

configPath_for_vehicle = "model/recognition.cfg"
weightsPath_for_vehicle = "model/recognition.weights"

JPEGDir = "JPEGImages"
AnnoDir = "numbers"

vehicle_net = cv2.dnn_Net

confThreshold = 0.8  # Confidence threshold
nmsThreshold = 0.6 # Non-maximum suppression threshold
import time
def load_model():
    global vehicle_net
    vehicle_net = cv2.dnn.readNetFromDarknet(configPath_for_vehicle, weightsPath_for_vehicle)

def list_images(path):
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]
    onlyfiles.sort()
    return onlyfiles

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
def Mbox(title, text, style):
    return ctypes.windll.user32.MessageBoxW(0, text, title, style)
def DetectionProcess(image, outs):
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:

            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > confThreshold:
                center_x = int(detection[0] * image.shape[1])
                center_y = int(detection[1] * image.shape[0])
                width = int(detection[2] * image.shape[1])
                height = int(detection[3] * image.shape[0])
                left = int(center_x - width / 2) + 2
                if (left < 0): left = 0
                top = int(center_y - height / 2) +2
                if (top < 0): top = 0
                right = left + int(width)
                if (right > image.shape[1]): right = image.shape[1] - 1
                bottom = top + int(height * 1.0)
                if (bottom > image.shape[0]): bottom = image.shape[0] - 1

                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, right, bottom])

    return boxes, classIds

    # indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # results = []
    # for i in indices:
    #     results.append(boxes[i])
    # return results, classIds
    # if len(indices) < 1: return [0, 0, 0, 0], -1
    #
    # max_conf = 0.0
    # max_index = 0
    # for i in indices:
    #     index = i
    #     if confidences[index] > max_conf:
    #         max_conf = confidences[index]
    #         max_index = index
    #
    # return boxes[max_index], classIds[max_index]


if __name__ == '__main__':
    file = open(r"result.txt","w+", encoding='utf-8')

    load_model()


    ptr = ""
    sim_card_numbers = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    # SOURCE_FOLDER = 'JPEGImages/'
    SOURCE_FOLDER = 'JPEGImages/'
    ls_images = list_images(SOURCE_FOLDER)
    for input_image in ls_images:
        startTime = time.time()
        inputFilename = os.path.join(SOURCE_FOLDER, input_image)
        img = cv2.imread(inputFilename)
        # cv2.imshow("original", img)
        # cv2.waitKey(0)

        crop_img = img
        blob = cv2.dnn.blobFromImage(crop_img, 1 / 255, (1280, 40), [0, 0, 0], 1,crop=False)
        # Sets the input to the network
        vehicle_net.setInput(blob, "data")
        # Runs the forward pass to get output of the output layers
        outs = vehicle_net.forward(getOutputsNames(vehicle_net))
        detection, classid = DetectionProcess(crop_img, outs)
        lbllist = []
        label = "region"
        for i in range(len(detection)):
            rect = detection[i]
            label = classid[i]
            if label == 10: label = 'a'
            if label == 11: label = 'b'
            lbllist.append([label, rect[0], rect[1], rect[2], rect[3]])
            # crop_img = cv2.rectangle(crop_img, (rect[0], rect[1]),(rect[2], rect[3]) , (0,0,255), 4)
            cv2.putText(crop_img, str(label), (rect[0], rect[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            # crop_img = cv2.putText(crop_img, (rect[0], rect[1]),(rect[2], rect[3]) , (0,0,255), 4)
        # for rect in detection:
        #     lbllist.append([label, rect[0], rect[1], rect[2], rect[3]])
        #     crop_img = cv2.rectangle(crop_img, (rect[0], rect[1]),(rect[2], rect[3]) , (0,0,255), 4)
        xmlstr = makeLabelling(JPEGDir, input_image, crop_img.shape, lbllist)
        aaa = input_image[0:-4]
        with open(os.path.join(AnnoDir, aaa + ".xml"), "w") as fout:
            fout.write(xmlstr)
            fout.close()

        scale_percent = 200 # percent of original size
        width = int(crop_img.shape[1] * scale_percent / 100)
        height = int(crop_img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
        # cv2.imshow("original", resized)
        # cv2.waitKey(0)
        endTime = time.time()
        print(f"Runtime of the program is {endTime - startTime}")
        print("\n")







import cv2
import os
import numpy as np
import ctypes  # An included library with Python install.


import time
from labellingEngine import makeLabelling

JPEGDir = r"./train_data/JPEGImages"
AnnoDir = r"./train_data/Annotations"

# Constants.
INPUT_WIDTH = 320
INPUT_HEIGHT = 320

recognition_classes = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "D", "H", "C", "S"]


vehicle_net = cv2.dnn_Net

confThreshold = 0.3  # Confidence threshold
nmsThreshold = 0.8 # Non-maximum suppression threshold
import time
def load_model():
	global vehicle_net
	#vehicle_net = cv2.dnn.readNet("./model/date_recognition.onnx")
	vehicle_net = cv2.dnn.readNet("./model/number.onnx")

def list_images(path):
	onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and f.endswith('.jpg')]
	onlyfiles.sort()
	return onlyfiles

def getOutputsNames(net):
	# Get the names of all the layers in the network
	layersNames = net.getLayerNames()
	# Get the names of the output layers, i.e. the layers with unconnected outputs
	return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
def Mbox(title, text, style):
	return ctypes.windll.user32.MessageBoxW(0, text, title, style)
def DetectionProcess(scale, outputs):
	outputs = np.array([cv2.transpose(outputs[0])])
	rows = outputs.shape[1]

	boxes = []
	scores = []
	class_ids = []

	for i in range(rows):
		classes_scores = outputs[0][i][4:]
		(minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
		if maxScore >= confThreshold:
			box = [
				(outputs[0][i][0] - (0.5 * outputs[0][i][2])) * scale, (outputs[0][i][1] - (0.5 * outputs[0][i][3])) * scale,
				(outputs[0][i][2]) * scale, (outputs[0][i][3]) * scale]
			boxes.append(box)
			scores.append(maxScore)
			class_ids.append(maxClassIndex)

	indices = cv2.dnn.NMSBoxes(boxes, scores, confThreshold, nmsThreshold)

	#return boxes, classIds
	if len(indices) < 1: return [], []
	boxes1 = []
	classIds1 = []
	for i in indices:
		index = i
		boxes1.append(boxes[index])
		classIds1.append(class_ids[index])

	return boxes1, classIds1


if __name__ == '__main__':
	file = open(r"result.txt","w+", encoding='utf-8')
   
	load_model()


	ptr = ""
	sim_card_numbers = []

	font = cv2.FONT_HERSHEY_SIMPLEX

	ls_images = list_images(JPEGDir)
	for input_image in ls_images:
		startTime = time.time()
		inputFilename = os.path.join(JPEGDir, input_image)
		img = cv2.imread(inputFilename)
		# cv2.imshow("original", img)
		# cv2.waitKey(0)
		[height, width, _] = img.shape
		length = max((height, width))
		image = np.zeros((length, length, 3), np.uint8)
		image[0:height, 0:width] = img
		scale = length / INPUT_WIDTH

		crop_img = img
		# Create a 4D blob from a frame.
		blob = cv2.dnn.blobFromImage(image, 1/255,  (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

		# Sets the input to the network.
		vehicle_net.setInput(blob)

		# Run the forward pass to get output of the output layers.
		outputs = vehicle_net.forward()

		detection, classid = DetectionProcess(scale, outputs)

		lbllist = []
		label = "region"
		print(len(detection))
		if len(detection) == 0: continue
		for i in range(len(detection)):
			rect = detection[i]
			label = recognition_classes[classid[i]]
			print(label)
			lbllist.append([label, int(rect[0]), int(rect[1]), int(rect[0] + rect[2]), int(rect[1] + rect[3])])
			crop_img = cv2.rectangle(crop_img, (int(rect[0]), int(rect[1])),(int(rect[0] + rect[2]), int(rect[1] + rect[3])) , (0,0,255), 4)
		xmlstr = makeLabelling(JPEGDir, input_image, crop_img.shape, lbllist)
		aaa = input_image[0:-4]
		with open(os.path.join(AnnoDir, aaa + ".xml"), "w") as fout:
			fout.write(xmlstr)
			fout.close()

		scale_percent = 100 # percent of original size
		width = int(crop_img.shape[1] * scale_percent / 100)
		height = int(crop_img.shape[0] * scale_percent / 100)
		dim = (width, height)
		resized = cv2.resize(crop_img, dim, interpolation = cv2.INTER_AREA)
		cv2.imshow("original", resized)
		cv2.waitKey(1)
		endTime = time.time()
		print(f"Runtime of the program is {endTime - startTime}")
		print("\n")






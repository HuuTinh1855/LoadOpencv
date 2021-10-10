# YOLO object detection
import cv2 as cv
import numpy as np
import time

img = cv.imread('images/img_low_frame5.jpg')
cv.imshow('window',  img)
cv.waitKey(1)

# Give the configuration and weight files for the model and load the network.
# net = cv.dnn.readNetFromDarknet('cfg/csdarknet53-omega.cfg', 'backup/csdarknet53-omega_1000s.weights')
net = cv.dnn.readNetFromDarknet('cfg/darknet53.cfg', 'backup/darknet53_1000.weights')
# net = cv.dnn.readNetFromDarknet('cfg/alexnet.cfg', 'backup/alexnet_last.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
class_names =["Low", "High"]

blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)


# set the input blob for the neural network
net.setInput(blob)
# forward pass image blog through the model
outputs = net.forward()

final_outputs = outputs[0]
# make all the outputs 1D
final_outputs = final_outputs.reshape(2, 1)
# get the class label
label_id = np.argmax(final_outputs)
# convert the output scores to softmax probabilities
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
# get the final highest probability
final_prob = np.max(probs) * 100.
# map the max confidence to the class label names
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}"
# put the class name text on top of the image
cv.putText(img, out_text, (25, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv.imshow('Image', img)
cv.waitKey(0)
cv.imwrite('result_image.jpg', img)
cv.destroyAllWindows()
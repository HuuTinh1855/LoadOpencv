import sys
import time
import os

from trafficui import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
import cv2 as cv
import numpy as np
from torchvision import models
import torch.nn as nn
import torch
from torch.autograd import Variable


class Traffic:
    def __init__(self):
        "Khởi tạo và chạy giao diện"
        app = QtWidgets.QApplication(sys.argv)
        main_windows = QtWidgets.QMainWindow()
        main_windows.setWindowIcon(QtGui.QIcon('icons/icon_traffic.png'))
        self.ui = Ui_MainWindow()
        self.ui.setupUi(main_windows)
        self.set_ui()
        self.net = {}
        self.loadModel()
        mssbox = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Information, "Crowd Detection Server",
                                       "Do you want to exit ?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("icons/icon_traffic.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        mssbox.setWindowIcon(icon)
        while True:
            main_windows.show()
            if app.exec_() == 0:
                if mssbox.exec() == QtWidgets.QMessageBox.No:
                    continue
                else:
                    break
            else:
                break
        sys.exit(0)

    def set_ui(self):
        self.ui.btnOpenFile.clicked.connect(self.btnOpenFile)
        self.ui.cbModel.currentTextChanged.connect(self.loadModel)

    def btnOpenFile(self):
        dlg = QFileDialog(directory='images')
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilters(["Image files (*.png *.xpm *.jpg)"])
        if dlg.exec_():
            filename = dlg.selectedFiles()[0]
            img = cv.imread(filename)
            if self.ui.cbModel.currentText()=="Darknet":
                img = self.predictDarknet(img)
                cv.imwrite('result/'+filename.split('/')[-1], img)
                img = QtGui.QPixmap('result/'+filename.split('/')[-1])
                img = img.scaled(self.ui.lbimage.width(), self.ui.lbimage.height())
                self.ui.lbimage.setPixmap(img)
            else:
                self.predictPytorch(img)
    def predictDarknet(self, img):
        class_names = ["Low", "High"]
        blob = cv.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # set the input blob for the neural network
        self.net.setInput(blob)
        # forward pass image blog through the model
        outputs = self.net.forward()

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
        return img

    def loadModel(self):
        if self.ui.cbModel.currentText() == "Darknet":
            self.net = cv.dnn.readNetFromDarknet('cfg/csdarknet53-omega.cfg', 'backup/csdarknet53-omega_1000s.weights')
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
        else:

            # define the directory for further converted model save
            onnx_model_path ="backup"
            # define the name of further converted model
            onnx_model_name = "alexnet.onnx"
            # create directory for further converted model
            os.makedirs(onnx_model_path, exist_ok=True)
            # get full path to the converted model
            full_model_path = os.path.join(onnx_model_path, onnx_model_name)
            if os.path.exists(full_model_path)==False:
                original_model = models.alexnet(pretrained=False)
                self.set_parameter_requires_grad(original_model, True)
                num_ftrs = original_model.classifier[6].in_features
                original_model.classifier[6] = nn.Linear(num_ftrs, 2)

                original_model.load_state_dict(
                    torch.load('backup/alexnet_13_best.pt', map_location=torch.device('cpu')))
                original_model.eval()
                # generate model input
                generated_input = Variable(
                    torch.randn(1, 3, 224, 224)
                )
                # model export into ONNX format
                torch.onnx.export(
                    original_model,
                    generated_input,
                    full_model_path,
                    verbose=True,
                    input_names=["input"],
                    output_names=["output"],
                    opset_version=11
                )
            # read converted .onnx model with OpenCV API
            self.net = cv.dnn.readNetFromONNX(full_model_path)
            # self.net = cv.dnn.readNetFromTorch("backup/alexnet_50.pt")
            self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def predictPytorch(self, img):
        class_names = ["Low", "High"]
        # read the image
        input_img = img.astype(np.float32)
        input_img = cv.resize(input_img, (256, 256))
        # define preprocess parameters
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        scale = 1 / 255.0
        std = [0.229, 0.224, 0.225]
        # prepare input blob to fit the model input:
        # 1. subtract mean
        # 2. scale to set pixel values from 0 to 1
        input_blob = cv.dnn.blobFromImage(
            image=input_img,
            scalefactor=scale,
            size=(224, 224),  # img target size
            mean=mean,
            swapRB=True,  # BGR -> RGB
            crop=True  # center crop
        )
        # 3. divide by std
        # input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

        # set OpenCV DNN input
        self.net.setInput(input_blob)
        # OpenCV DNN inference
        out = self.net.forward()
        print("OpenCV DNN prediction: \n")
        print("* shape: ", out.shape)
        # get the predicted class ID
        imagenet_class_id = np.argmax(out)
        # get confidence
        confidence = out[0][imagenet_class_id]
        print("* class ID: {}, label: {}".format(imagenet_class_id, class_names[imagenet_class_id]))
        print("* confidence: {:.4f}".format(confidence))

    def predictVideo(self, filename):
        # capture the video
        cap = cv.VideoCapture(filename)
        # detect objects in each frame of the video
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                image = frame
                image_height, image_width, _ = image.shape
                # create blob from image
                blob = cv.dnn.blobFromImage(image=image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
                # start time to calculate FPS
                start = time.time()
                self.net.setInput(blob)
                output = self.net.forward()
                # end time after detection
                end = time.time()
                # calculate the FPS for current frame detection
                fps = 1 / (end - start)
                cv.imshow('image', image)
                if cv.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv.destroyAllWindows()
if __name__ == "__main__":
    traffic = Traffic()

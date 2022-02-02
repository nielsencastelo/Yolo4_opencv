# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 11:07:26 2021

@author: niels
"""

import cv2
import time
import numpy as np
import imutils




# Parametros da leitura da Rede Neural
th1 = 0.1
th2 = 0.2
class_names = []

with open("obj.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]


# Load Yolo
net = cv2.dnn.readNet("yolov4.weights", "yolov4-custom.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(608,608), scale=1/255)

# Parametros do video
totalFrames = 0
skip_frames = 1

dimensao = 640

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output_real_time.avi',fourcc, 20.0, (640,480))

cap = cv2.VideoCapture('rigo.mp4')

ok, frame = cap.read()
if not ok:
    print('Não é possível ler o arquivo de vídeo')
    sys.exit()

while True:
    ret, frame = cap.read()
    
    if not (ret):
            print('Erro no frame')
            st = time.time()
            cap = cv2.VideoCapture(url)
            print("tempo perdido devido à inicialização cam  : ", time.time()-st)
            continue
    
    if ret == True:
        if totalFrames % skip_frames == 0:
        
            #frame = imutils.resize(frame, width=dimensao)
            
            start = time.time() # para calcular o FPS
            classes, scores, boxes = model.detect(frame, th1, th2)
            end = time.time()
    
            for (classesId, score, box) in zip(classes, scores, boxes):
                label = f"{class_names[classesId[0]]} {score}"
                
                cv2.rectangle(frame, box, (255,0,0), 2)
                
                cv2.putText(frame, label, (box[0],box[1] - 10), 0, 0.5, (255,0,0), 2)
        
    
            fps = f"fps { round((1/(end-start)), 2) }"
            
            cv2.putText(frame, fps, (20,20), 0, 0.5, (255,0,0), 2)
            out.write(frame)
            cv2.imshow('Deteccao CAM ', frame)
            
    totalFrames += 1
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
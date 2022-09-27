#!/usr/bin/python3

import sys
import dlib
import cv2
import os
import re
import json
from pylab import *
from PIL import Image, ImageChops, ImageEnhance

a = sys.argv[1]
b = sys.argv[2]
if a == 'train':
    train_frame_folder = '/run/media/kk/BaKa/Celeb-DF-v2_2'+b
else:
    train_frame_folder = '/run/media/kk/BaKa/Celeb-DF-v2_2/test_video'+b
#with open(os.path.join(train_frame_folder, 'metadata.json'), 'r') as file:
#    data = json.load(file)
list_of_train_data = [f for f in os.listdir(train_frame_folder) if f.endswith('.mp4')]
detector = dlib.get_frontal_face_detector()
for vid in list_of_train_data:
    count = 0
    cap = cv2.VideoCapture(os.path.join(train_frame_folder, vid))
    frameRate = cap.get(5)
    while cap.isOpened():
        frameId = cap.get(1)
        ret, frame = cap.read()
        if ret != True:
            break
        if frameId % ((int(frameRate)+1)*1) == 0:
            face_rects, scores, idx = detector.run(frame, 0)
            for i, d in enumerate(face_rects):
                x1 = d.left()
                y1 = d.top()
                x2 = d.right()
                y2 = d.bottom()
                crop_img = frame[y1:y2, x1:x2]

                if b == '/Celeb-synthesis':
                    cv2.imwrite('/run/media/kk/BaKa/Celeb-DF-v2_2/datasheet/'+a+'/fake/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (150, 150),fx=0,fy=0, interpolation = cv2.INTER_CUBIC))
                else:
                    cv2.imwrite('/run/media/kk/BaKa/Celeb-DF-v2_2/datasheet/'+a+'/real/'+vid.split('.')[0]+'_'+str(count)+'.png', cv2.resize(crop_img, (150, 150),fx=0,fy=0, interpolation = cv2.INTER_CUBIC))

                count+=1





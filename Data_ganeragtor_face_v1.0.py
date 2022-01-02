# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 13:06:34 2021

@author: Clark
"""
import cv2
import numpy as np
import os
#import tensorflow as tf
import cvlib as cv
import pandas as pd
import matplotlib.pyplot as plt

###################################################################

"""
try out all know resolutions
"""
# url = "https://en.wikipedia.org/wiki/List_of_common_resolutions"
# table = pd.read_html(url)[0]
# table.columns = table.columns.droplevel()

# cap = cv2.VideoCapture(0)
# resolutions = {}

# for index, row in table[["W", "H"]].iterrows():
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, row["W"])
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, row["H"])
#     width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     resolutions[str(width)+"x"+str(height)] = "OK"

# print(resolutions)

# {'160.0x120.0': 'OK', 
#  '320.0x180.0': 'OK', 
#  '320.0x240.0': 'OK', 
#  '424.0x240.0': 'OK', 
#  '352.0x288.0': 'OK', 
#  '640.0x360.0': 'OK', 
#  '640.0x480.0': 'OK', 
#  '848.0x480.0': 'OK', 
#  '960.0x540.0': 'OK', 
#  '1280.0x720.0': 'OK'}

###################################################################


base_dir = './data/'   
target_cnt = 30   
cnt = 0
result =pd.DataFrame()

# opencv haarcascade api 사용의 경우
#face_classifier = cv2.CascadeClassifier('./opencv-4.x/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml')

name = input("Insert User Class Name(Only Alphabet):")
id = input("Insert User Class Id(Non-Duplicate number):")

dir = os.path.join(base_dir, name+'_'+ id)
if not os.path.exists(dir):
    os.makedirs(dir)

cap = cv2.VideoCapture(0)

#yolo input size : 416 but my cam resolution oonly 424*240
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 424)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

# cv2.namedWindow('image')
# cv2.resizeWindow(winname='image', width=416, height=416)

if not cap.isOpened():
    print('No Frame')
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        img = frame.copy()
        #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #gray = img.copy()
        #faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        #cvlib dnn module 사용
        faces, confidence = cv.detect_face(frame)        
        
        if len(faces) == 1:
            print(f'faces coordi : {faces}')
            #result_coordi.extend(faces)
            #(x,y,w,h) = faces[0]
            (startX, startY, endX, endY) = faces[0]
            endX = endX-10
            
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 1)
#             face = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (startX,startY), (endX, endY), (0,255,0), 1)
            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (200, 200))
            file_name_path = os.path.join(dir,  str(cnt) + '.jpg')
            cv2.imwrite(file_name_path, face)
            
            file_name_raw_path = os.path.join(dir,  str(cnt) + '_raw.jpg')
            img = cv2.resize(img, (424, 240))
            cv2.imwrite(file_name_raw_path, img)
            
            dx = 424/640 
            dy = 240/480
            
            filename = list([file_name_raw_path])
            region_shape_attributes = list([{"name":"rect","xmin":int(startX*dx),"ymin":int(startY*dy),"xmax":int(endX*dx),"ymax":int(endY*dy)}])
            #print(str(region_shape_attributes))
            region_attributes = list([{"object":id}])
            
            
            temp_df = pd.DataFrame(data = [x for x in zip(filename,region_shape_attributes,region_attributes)], columns=['filename', 'region_shape_attributes', 'region_attributes'])
            result = pd.concat([result,temp_df], ignore_index=True)
            
            cv2.putText(frame, str(cnt), (startX,startY), cv2.FONT_HERSHEY_COMPLEX, \
                             1, (0,255,0), 2)
            cnt+=1
        else:
            if len(faces) == 0 :
                msg = "no face."
            elif len(faces) > 1:
                msg = "too many face."
            cv2.putText(frame, msg, (10, 50), cv2.FONT_HERSHEY_DUPLEX, \
                            1, (0,0,255))
        cv2.imshow('face record', frame)
        if cv2.waitKey(1) == 27 or cnt == target_cnt:
            break
cap.release()
cv2.destroyAllWindows()

#result.to_csv(os.path.join(base_dir,'{}-result.csv'.format(name+'_'+ id)), index=False)

if not os.path.exists(os.path.join(base_dir,'result.csv')):
    result.to_csv(os.path.join(base_dir,'result.csv'), index=False)
    print("result",result)
else:
    df = pd.read_csv(os.path.join(base_dir,'result.csv'))
    df = pd.concat([df,result], ignore_index=True)
    df.to_csv(os.path.join(base_dir,'result.csv'), index=False)
    print("result",df)

print("Collecting Samples Completed.")

"""
test
"""
# test_df = pd.read_csv(os.path.join(base_dir,'result.csv'))
# img_path = test_df['filename'][0]
# color = cv2.imread(img_path, cv2.IMREAD_COLOR)
# print('image shape:',color.shape)
# face = eval(test_df['region_shape_attributes'][0])

# cv2.rectangle(color, (face['xmin'],face['ymin']), (face['xmax'], face['ymax']), (0,255,0), 1)
# cv2.imshow('face test', color)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

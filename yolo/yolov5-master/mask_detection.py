import os
import cv2
import torch
import numpy as np
from models.common import DetectMultiBackend
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

## Load model

model = torch.hub.load(os.getcwd(), 'custom', path='best.onnx', source='local')

## Setting inference attributes

model.conf = 0.5            # NMS confidence threshold
model.iou = 0.45            # NMS IoU threshold

## Load testing image

name = model.names
print(name)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
        print("Cannot open camera")
        exit()

while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = frame
        img_RGB = cv2.cvtColor( img , cv2.COLOR_BGR2RGB )

        original_Size = img_RGB.shape
        img_resize = cv2.resize( img_RGB , (640,640) )
        results = model(img_resize)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        class_name = ['Mask','No Mask','Bad Mask']

        scale_x = original_Size[1]/640
        scale_y = original_Size[0]/640

        New_BBox = []
        for point in results.xyxy[0].cpu().numpy():
            New_BBox.append( [int(point[0]*scale_x),int(point[1]*scale_y),int(point[2]*scale_x),int(point[3]*scale_y),int(point[5])] )

        img_show = img.copy()
        for point_revers in New_BBox :
            cv2.rectangle( img_show , (point_revers[0],point_revers[1]) ,(point_revers[2],point_revers[3]) ,colors[point_revers[4]] ,2 )
            cv2.putText(img_show, class_name[point_revers[4]], (int(point_revers[0]*scale_x)+10,int(point_revers[1]*scale_y)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, colors[point_revers[4]], 2, cv2.LINE_AA)

        cv2.imshow('im',img_show)
        if cv2.waitKey(10) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          cap.release()
          break
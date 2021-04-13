import cv2
import time
import imutils
import argparse
import numpy as np

from imutils.video import FPS
from imutils.video import VideoStream

use_gpu= True
#Constructing Argument Parse to input from Command Line
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type = float, default = 0.5)
args = vars(ap.parse_args())

#Initialize Objects and corresponding colors which the model can detect
labels = ["background", "aeroplane", "bicycle", "bird", 
"boat","bottle", "bus", "car", "cat", "chair", "cow", 
"diningtable","dog", "horse", "motorbike", "person", "pottedplant", 
"sheep","sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))

#Loading Caffe Model
print('[Status] Loading Model...')
proto ='Caffe/SSD_MobileNet_prototxt.txt'
model = 'Caffe/SSD_MobileNet.caffemodel'
nn = cv2.dnn.readNetFromCaffe(proto, model)
if (use_gpu == True ):
	# set CUDA as the preferable backend and target
	print("[INFO] setting preferable backend and target to CUDA...")
	nn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	nn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = cv2.VideoCapture(0)
vs.set(3,320)
vs.set(4,288)
time.sleep(2.0)
fps = FPS().start()

#Loop Video Stream
while True:

    #Resize Frame to 400 pixels
    _,frame = vs.read()
    frame = cv2.resize(frame,dsize=(400,300))
    #(h, w) = frame.shape[:2]
    h = frame.shape[0]
    w =frame.shape[1]
    #Converting Frame to Blob
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 
    	0.007843, (300, 300), 127.5)

    #Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()


    #Loop over the detections
    for i in np.arange(0, detections.shape[2]):

	#Extracting the confidence of predictions
        confidence = detections[0, 0, i, 2]

        #Filtering out weak predictions
        if confidence > args["confidence"]:
            
            #Extracting the index of the labels from the detection
            #Computing the (x,y) - coordinates of the bounding box        
            idx = int(detections[0, 0, i, 1])
            print(idx)

            #Extracting bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            #Drawing the prediction and bounding box
            label = "{}: {:.2f}%".format(labels[idx], confidence * 100)
            if(idx== 15 and confidence >0.7):
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, "human", (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    
    fps.update()

fps.stop()

print("[Info] Elapsed time: {:.2f}".format(fps.elapsed()))
print("[Info] Approximate FPS:  {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()

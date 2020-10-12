import cv2

# image =cv2.imread("image/cycle.jpg")
thres = 0.45
cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,720)
# cap.set(10,150)


classNames = []
classfile = 'coconames.txt'
with open(classfile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v1_coco.pbtxt'
weightspath ='frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightspath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5 , 127.5 ,127.5))
net.setInputSwapRB(True)
while True:
    success,image = cap.read()
    classIds, confs, bbox = net.detect(image,confThreshold=thres)
    print(classIds,bbox)
    if len(classIds) != 0:
        for classId , confidence , box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(image,box,color=(255,0,0),thickness=3)
            cv2.putText(image,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),thickness=1)

            cv2.putText(image, str(round(confidence*150,2)), (box[0] + 200, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), thickness=1)

    cv2.imshow("output",image)

    cv2.waitKey(1)

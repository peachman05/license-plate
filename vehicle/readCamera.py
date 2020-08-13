import cv2
from darknet.python.darknet import detect
import darknet.python.darknet as dn
import zmq
import base64
import json
import pickle

context = zmq.Context()
print("Connecting server ... ")
socket = context.socket(zmq.PUSH)
socket.connect("tcp://streamer_detector:5561")

weights = './darknet/yolov4.weights'
netcfg  = './darknet/cfg/yolov4.cfg'
data = './darknet/cfg/coco.data'
thresh = 0.5
net  = dn.load_net(netcfg.encode('utf-8'), weights.encode('utf-8'), 0)
meta = dn.load_meta(data.encode('utf-8'))

accept = ['car','motorbike','truck','bus']
# frame_id = 0
cam = context.socket(zmq.PULL)
cam.connect("tcp://streamer_camera:5560")
# print('*************************ddd******************')
while True:
    # print('*******************************************')
    data = cam.recv_pyobj()
    frame = data['frame']
    frame_id = data['frame_id']
    detected_objects = detect(net, meta, frame, thresh=thresh)
    metas=[]
    for obj,confidence,rect in detected_objects:
        obj = obj.decode('utf-8')
        # print(confidence,obj,rect)
        if obj in accept:
            x,y,w,h = rect
            x = x-(w/2)
            y = y-(h/2)
            detected = {
                'cls':obj,
                'x':x,
                'y':y,
                'w':w,
                'h':h
            }
            metas.append(detected)
    # ret, frame = cv2.imencode('.png', frame)
    # frame = pickle.dumps(frame, 0)
    data = {
        'frame_id':frame_id,
        'frame':frame,
        'meta':metas
    }
    socket.send_pyobj(data)
    # frame_id+=1
cap.release()
cv2.destroyAllWindows()
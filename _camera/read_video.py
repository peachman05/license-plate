import cv2
from utils.camera import camera
import zmq
import time

context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
# zmq_socket.connect("tcp://127.0.0.1:5559")
zmq_socket.connect("tcp://titan.local:5559")


# frame_id = 0
# camera_id = 0
# while(1):
#     frame = cam.get_frame(1)
#     cv2.imshow("Feed",frame)

#     data = {
#         'camera_id':camera_id,
#         'frame_id':frame_id,
#         'frame':frame
#     }
#     zmq_socket.send_pyobj(data)
#     key = cv2.waitKey(1)
#     if key == 13: #13 is the Enter Key
#         break
# cv2.destroyAllWindows()     

# cam.end()


import numpy as np
import cv2

cap = cv2.VideoCapture('/home/peachman/Documents/Video/data1440 2k 30fps.mp4')
# cap = cv2.VideoCapture('/home/msi/Documents/Video/data1440 2k 60fps.mp4')
# cap = cv2.VideoCapture('/home/msi/Documents/Video/data2160 4k 30fps.mp4')
# start_frame = 5000
start_frame = 200#1100 #1200 #200
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('total :', total)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_id = start_frame
count = 0
count2 = 0
camera_id = 0
frame_list = [200, 300]
jump_rate = 10
while(True):
    # Capture frame-by-frame
    if count < 10:
        ret, frame = cap.read()
        if frame_id % jump_rate == 0:
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            # ret, frame = cap.read()
            # frame = cv2.resize(frame,None,fx=1,fy=1)
            if ret:
                data = {
                    'camera_id':camera_id,
                    'frame_id':frame_id,
                    'frame':frame
                }
                zmq_socket.send_pyobj(data)
                
                # Display the resulting frame
                frame = cv2.resize(frame,None,fx=0.4,fy=0.4)
                cv2.imshow('input',frame)
                print(count2, frame_id)
                
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            count += 1
            count2 += 1
            
        frame_id += 1
        
    else:
        count = 0
        print('sleep 60 seconds')
        time.sleep(10)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
import cv2
from utils.camera import camera
import zmq

context = zmq.Context()
zmq_socket = context.socket(zmq.PUSH)
zmq_socket.connect("tcp://127.0.0.1:5559")

# RTSP_PATH = "rtsp://admin:qwer1234@192.168.88.251:554/Streaming/channels/1"
# RTSP_PATH = "rtmp://172.30.71.151/live/test"
# RTSP_PATH = "rtmp://172.30.71.117/live/test"
RTSP_PATH = "rtmp://172.17.0.1/live/test"
cam = camera(RTSP_PATH)
print('testtt')
print(f"Camera is alive?: {cam.p.is_alive()}")

frame_id = 0
camera_id = 0
while(1):
    frame = cam.get_frame(0.65)
    cv2.imshow("Feed",frame)

    data = {
        'camera_id':camera_id,
        'frame_id':frame_id,
        'frame':frame
    }
    zmq_socket.send_pyobj(data)
    key = cv2.waitKey(1)
    if key == 13: #13 is the Enter Key
        break
cv2.destroyAllWindows()     

cam.end()
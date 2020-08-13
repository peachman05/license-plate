import cv2
import numpy as np


fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = int(2560/2)
height = int(1440/2)
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))

frame = np.ones((height,width,3)).astype('uint8')
for i in range(300):
    out.write(frame)

out.release()
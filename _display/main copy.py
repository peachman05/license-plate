import cv2
import zmq
import numpy as np
from PIL import Image,ImageDraw,ImageFont


context = zmq.Context()
zmq_socket = context.socket(zmq.PULL)
zmq_socket.connect("tcp://127.0.0.1:5564")

fnt = ImageFont.truetype("Sarun's ThangLuang.ttf", 20)
fnt2 = ImageFont.truetype("Sarun's ThangLuang.ttf", 16)
while(1):
    data = zmq_socket.recv_pyobj()
    print(data.keys())
    frame = data['frame']
    # plate = data['plate']
    # result = data['result']
    # cv2.imshow('frame',frame)
    # cv2.imshow('plate',plate)
    # print(frame.shape)
    # frame = data['car'].astype(np.uint8)
    # print(frame.shape, type(frame))
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)   
    
    for vehicle in data['vehicles']:
        x1 = vehicle['x']
        y1 = vehicle['y']
        x2 = x1+vehicle['w']
        y2 = y1+vehicle['h']
        
        
        draw.rectangle(((x1, y1), (x2, y2)), outline=(0,255,0), width=4)

        if 'plate' in vehicle.keys():
            result = vehicle['plate']['result']
            draw.text(((x1+x2)/2, y1+20), result, font=fnt, fill=(0,255,0,128))
            draw.text(((x1+x2)/2, y1+45), "ปทุมธานี", font=fnt2, fill=(0,255,255,255))
            draw.text(((x1+x2)/2, y1+65), "โตโยต้า ชมพู", font=fnt2, fill=(255,255,255,255))

        
        # temp = data['test'][0]
        # x1 = temp[0]
        # y1 = temp[1]
        # x2 = x1+temp[2]
        # y2 = y1+temp[3]
        # draw.rectangle(((x1, y1), (x2, y2)), outline=(0,255,0), width=4)
    
    frame2 = np.array(img)
    print(frame2.shape)
    resize_img = cv2.resize(frame2,None,fx=1,fy=1)
    cv2.imshow('Output',resize_img)

    print('id:', data['frame_id'])   

    # cv2.imshow('car',data['car'])
    # print('Len: ',len(data['plate']))
    # cv2.imshow('plate',data['plate'][0])
    # img.show()


    # print(result)
    # print(data['meta'])

    # print(data['test'])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    sorted(lis, key = lambda i: i['age'],reverse=True) 
    # cv2.waitKey(1)
cv2.destroyAllWindows()     
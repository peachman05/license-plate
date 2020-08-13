import cv2
import zmq
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import json
import pickle

context = zmq.Context()
zmq_socket = context.socket(zmq.PULL)
# zmq_socket.connect("tcp://127.0.0.1:5564")
zmq_socket.connect("tcp://titan.local:5564")

fnt = ImageFont.truetype("Sarun's ThangLuang.ttf", 40)
fnt2 = ImageFont.truetype("Sarun's ThangLuang.ttf", 30)

images_buffer = []
buffer_size = 1
buffer_send = 1

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
width = int(2560/2)
height = int(1440/2)
width2 = 2560
height2 = 1440
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width,height))
outfull = cv2.VideoWriter('outputFull.mp4', fourcc, 10.0, (width2,height2))

Exit = False
count = 0

dict_list = []
while(1):
    for i in range(buffer_size-len(images_buffer)):
        data = zmq_socket.recv_pyobj()
        count += 1
        
        # print(data.keys())
        frame = data['frame']
        # print(frame.shape)

        data_temp = data.copy()
        del data_temp['frame']
        dict_list.append(data_temp)
        outfull.write(frame)

        # plate = data['plate']
        # result = data['result']
        # cv2.imshow('frame',frame)
        # cv2.imshow('plate',plate)
        # print(frame.shape)
        # frame = data['car'].astype(np.uint8)
        # print(frame.shape, type(frame))
        # frame = cv2.resize(frame,None,fx=0.7,fy=0.7)
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img, 'RGBA')   
        print('-----------')
        print('len :',len(data['vehicles']))
        for j, vehicle in enumerate(data['vehicles']):
            x1 = vehicle['x']
            y1 = vehicle['y']
            x2 = x1+vehicle['w']
            y2 = y1+vehicle['h']
            print('x={}, y={}'.format(int(x1),int(y1)))
            
            draw.rectangle(((x1, y1), (x2, y2)), outline=(0,255,0), width=2)
            draw.text((x1, y1-50), 'x={}, y={}'.format(int(x1),int(y1)), font=fnt, fill=(0,255,0,20))

            x = frame.shape
            w1 , h1 = x[0],x[1]

            q1 = vehicle['w']*vehicle['h']
            q2 = w1*h1
            # print(q1/q2)
            # print(vehicle['prob'])
            if q1/q2 > 0.000: #0.03:
                draw.text(((x1+x2)/2, y1-2), str(vehicle['prob']), font=fnt, fill=(255,0,0,255))
                if 'plate' in vehicle.keys():
                    result = vehicle['plate']['result']
                    province = vehicle['plate']['province_result']
                    province_prob = vehicle['plate']['province_prob']
                    # draw.rectangle(((x1, y1), (x2, y2)), fill=(0,255,0,20))
                    draw.rectangle(((x1, y1), (x2, y2)))
                    draw.text(((x1+x2)/2, y1+20), result, font=fnt, fill=(0,255,0,20))
                    draw.text(((x1+x2)/2, y1+50), province+str(province_prob), font=fnt2, fill=(0,255,255,255))                   

                    pts = vehicle['plate']['ptspx']
                    for i in range(4):
                        pt1 = tuple(pts[:,i].astype(int).tolist())
                        pt2 = tuple(pts[:,(i+1)%4].astype(int).tolist())
                        print(pt1,pt2)
                        draw.line([pt1, pt2],fill=(0,255,255), width= 15)
                        # cv2.line(I,pt1,pt2,color,thickness)

                    # cv2.imshow(str(j),vehicle['plate']['image'])
                    # cv2.imshow('car',vehicle['plate']['car_image'])
                    # print(vehicle['plate']['image'].shape)
                    # # cv2.waitKey(500)

                if 'car' in vehicle.keys() and vehicle['car']['car_prob'] > 0.2:
                    car_info = vehicle['car']['car_info']
                    car_prob = vehicle['car']['car_prob']
                    draw.text(((x1+x2)/2, y1+70), car_info, font=fnt2, fill=(255,255,255,255))
                    draw.text(((x1+x2)/2, y1+100), str(car_prob), font=fnt2, fill=(255,255,255,255))

            
            # temp = data['test'][0]
            # x1 = temp[0]
            # y1 = temp[1]
            # x2 = x1+temp[2]
            # y2 = y1+temp[3]
            # draw.rectangle(((x1, y1), (x2, y2)), outline=(0,255,0), width=4)
        print(count)    
        
        # print(frame2.shape)

        frame2 = np.array(img)        
        resize_img = cv2.resize(frame2,None,fx=0.5,fy=0.5)
        out_image = cv2.resize(frame2,(width,height))
        out.write(out_image)
        # cv2.imshow('Output',resize_img)
        print('{}: get {} frame'.format(count, i),end='\r')
        # print('get id:', data['frame_id'])   

        images_buffer.append({'frame_id': data['frame_id'], 'image': out_image})

        cv2.imshow('Output',out_image)
        cv2.waitKey(90)

        

        # cv2.destroyAllWindows() 

      

    # sort_frames = sorted(images_buffer, key = lambda i: i['frame_id']) 
    # for dictObj in sort_frames[:buffer_send]:
    #     # print('id: ', dictObj['frame_id'])
        
    #     cv2.imshow('Output',dictObj['image'])
    #     # print(data['test'])
    #     if cv2.waitKey(5000) & 0xFF == ord('q'):
    #         Exit = True
    #         break
    
    # cv2.destroyAllWindows() 
    
    if Exit:
        break
    
    images_buffer[:] = images_buffer[buffer_send:]
    # print(len(images_buffer))

out.release()
# outfull.release()


with open('frameInfo.pickle', 'wb') as handle:
    pickle.dump(dict_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('frameInfo.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(dict_list == b)

    # cv2.waitKey(1)
cv2.destroyAllWindows()     
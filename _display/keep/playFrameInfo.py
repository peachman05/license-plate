import pickle
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont


fnt = ImageFont.truetype("Sarun's ThangLuang.ttf", 40)
fnt2 = ImageFont.truetype("Sarun's ThangLuang.ttf", 30)

with open('frameInfo.pickle', 'rb') as handle:
    frame_info = pickle.load(handle)

print(frame_info[0].keys())
# print(frame_info)

cap = cv2.VideoCapture('outputFull.mp4')
i = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    data = frame_info[i]

    # Numpy to Pillow
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, 'RGBA')   
    
    for vehicle in data['vehicles']:
        x1 = vehicle['x']
        y1 = vehicle['y']
        x2 = x1+vehicle['w']
        y2 = y1+vehicle['h']
        
        
        draw.rectangle(((x1, y1), (x2, y2)), outline=(0,255,0), width=2)

        x = frame.shape
        w1 , h1 = x[0],x[1]

        q1 = vehicle['w']*vehicle['h']
        q2 = w1*h1
        # print(q1/q2)
        if q1/q2 > 0.03: #0.03:
            if 'plate' in vehicle.keys():
                result = vehicle['plate']['result']
                province = vehicle['plate']['province_result']
                province_prob = vehicle['plate']['province_prob']
                # draw.rectangle(((x1, y1), (x2, y2)), fill=(0,255,0,20))
                draw.rectangle(((x1, y1), (x2, y2)))
                draw.text(((x1+x2)/2, y1+20), result, font=fnt, fill=(0,255,0,20))
                
                if province_prob > 0.99:
                    draw.text(((x1+x2)/2, y1+50), province+str(province_prob), font=fnt2, fill=(0,255,255,255))                   

            
            if 'car' in vehicle.keys() and vehicle['car']['car_prob'] > 0.9:
                car_info = vehicle['car']['car_info']
                car_prob = vehicle['car']['car_prob']
                draw.text(((x1+x2)/2, y1+70), car_info + str(car_prob), font=fnt2, fill=(255,255,255,255))
                # draw.text(((x1+x2)/2, y1+100), , font=fnt2, fill=(255,255,255,255))

    frame2 = np.array(img)        
    resize_img = cv2.resize(frame2,None,fx=0.6,fy=0.6)
    cv2.imshow('Output', resize_img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        break
    
    i += 1


cap.release()
cv2.destroyAllWindows()
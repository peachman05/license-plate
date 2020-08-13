# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow
import keras
from keras import backend as K
import numpy as np
import time
import cv2
import os, sys
import random
import glob
import zmq
# import matplotlib.pyplot as plt
import traceback
from src.alpr.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.alpr.utils 					import im2single,adjust_pts,crop_region
from src.alpr.keras_utils 			import load_model, detect_lp
from src.alpr.label 					import Shape, writeShapes, Label
from src.lprnet.utils import TextImageGenerator,encode_label,sparse_tuple_from,decode_sparse_tensor,\
                    decode_a_seq,small_basic_block,conv,report_accuracy,\
                    test_report,do_batch,get_test_model,predict_oneimg

# from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.autograd import Variable

def getProvinceModel():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(80,240,3)))
    # model.add(Conv2D(64, kernel_size=5, activation='relu', input_shape=(40,240,3)))
    model.add(MaxPooling2D((3, 3)))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D((3, 3)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    return model
    
# session = keras.backend.get_session()
# init = tf.global_variables_initializer()
# session.run(init)

# init = tf.global_variables_initializer()
# sess.run(init)



print(sys.getdefaultencoding())

### SET UP
gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.05)
config = tensorflow.ConfigProto( device_count = {'GPU': 1 , 'CPU': 16}, gpu_options=gpu_options ) 
config.gpu_options.allow_growth = True

''' INIT ALPR '''
wpod_sess = tf.Session(config=config) 
keras.backend.set_session(wpod_sess)
lp_threshold = .5
wpod_net_path = 'models/alpr/v2_2_backup'   #'dti-trained-model_final'
wpod_net = load_model(wpod_net_path)
BATCH_SIZE = 1
TRAIN_SIZE = 1000
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 1
ti = 'lprnet/train_lprnet/lpr_label1.csv'
vi = 'lprnet/train_lprnet/lpr_label1.csv'
img_size = [94, 24]
tl = None
vl = None
num_channels = 3
label_len = 7
lprNet_model = 'models/lprnet/v2/LPRtf3.ckpt-110000'
CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
         'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z',
         'ก', 'ข', 'ฃ', 'ค', 'ฅ', 'ฆ', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 
         'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 'ด', 'ต', 'ถ', 
         'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 
         'ย', 'ร', 'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 
         '_'
         ]
CHARS_DICT = {char:i for i, char in enumerate(CHARS)}
NUM_CHARS = len(CHARS)

''' PROVINCE '''
province_name = ['กรุงเทพมหานคร','สกลนคร','อุดรธานี']
province_model = getProvinceModel()
province_model.load_weights('PlateFull-435-0.97-1.00.h5')


'''INIT LPRNET'''
lprnet_graph = tf.Graph()
with lprnet_graph.as_default() as graph:
    global_step = tf.Variable(0, trainable=False)
    print(num_channels, label_len, BATCH_SIZE, img_size)
    logits, inputs, seq_len = get_test_model(num_channels, label_len,BATCH_SIZE, img_size,NUM_CHARS)
    logits = tf.transpose(logits, (1, 0, 2))
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
    lprnet_init_op = tf.global_variables_initializer()

lprnet_sess = tf.Session(graph=lprnet_graph, config=config)
lprnet_sess.run(lprnet_init_op)

with lprnet_graph.as_default() as g:
    with lprnet_sess.as_default() as sess:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        saver.restore(sess, lprNet_model)

print(K.get_session() == wpod_sess)


test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = torch.load('./models/resNet/dit_08_06.h5',  map_location='cuda:0')
model_ft = model_ft.to(device)
model_ft.eval()
f = open('./models/resNet/dit_08_06.txt','r')
class_name=f.readline().split(',')
f.close()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def predict_image(image):
    with torch.no_grad():
        image_tensor = test_transforms(image).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input = Variable(image_tensor)
        input = input.to(device)
        output = model_ft(input)
        result = output.data.cpu().numpy()
        prob_result = softmax(result[0])
        index = result.argmax()
        return class_name[index], prob_result[index]

'''Recv Data'''
context = zmq.Context()
zmq_socket = context.socket(zmq.PULL)
zmq_socket.connect("tcp://streamer_detector:5562")

zmq_send = context.socket(zmq.PUSH)
zmq_send.connect("tcp://forwarder:5563")

start_time = time.time()
print ('Searching for vehicles using LP using WPOD-NET...')
while True:

    data = zmq_socket.recv_pyobj()
    Iorig = data['frame']                       
    frame_id = data['frame_id']

    # vehicle dict for output
    vehicle = []

    for index,obj in enumerate(data['meta']):
        name = obj['cls']       
        WH = np.array(Iorig.shape[1::-1],dtype=float)
        # print((obj['x'],obj['y'],obj['w'],obj['h']))
        car_frame = Iorig[int(obj['x']):int(obj['x']+obj['w']), int(obj['y']):int(obj['y']+obj['h'])]
        print(type(car_frame))
        try:
            car_frame = cv2.cvtColor(car_frame, cv2.COLOR_BGR2RGB)
        except Exception:
            continue

        car_frame = Image.fromarray(car_frame)
        car_info, car_prob = predict_image(car_frame)
        cx,cy,w,h = (np.array((obj['x'],obj['y'],obj['w'],obj['h']))/np.concatenate( (WH,WH) )).tolist()
        tl = np.array([cx, cy])
        br = np.array([cx + w, cy + h])
        label = Label(0,tl,br)
        # print(label)
        Icar = crop_region(Iorig,label)
        if np.any(Icar) == 0:
            continue
        try:
            Ivehicle = Icar.astype('uint8')
        except Exception as e:
            continue
            # print(e)
        ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
        side  = int(ratio*288.)
        bound_dim = min(side + (side%(2**4)),608)
        print( "\t\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

        start_time_alpr = time.time()
        Llp,LlpImgs,ptsh,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
        print('ALPR time: ', time.time()-start_time_alpr)

        # print( "\t\t\tNo. LP: %d" % len(LlpImgs))

        ## For packing new vehicle
        vehicle_detail = obj.copy()

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)
            # print("S : ", Llp[0].pts)
            # print("S : ", Shape(Llp[0].pts))
            # print((Ilp*255.).astype('uint8'))

            #### Calculate plate bbox
            ptspx = ptsh.copy()
            ptspx[0] +=  tl[0] # Add x 
            ptspx[1] +=  tl[1] # Add y

            Ilp = cv2.resize(Ilp,(94,24))
            start_time_ocr= time.time()
            result = predict_oneimg(Ilp*255,inputs,seq_len,lprnet_sess,decoded,CHARS)
            # result = 'test'
            ### Predict Province 
            license_img = np.array([LlpImgs[0]])
            # print('License range:',np.max(license_img),np.min(license_img))
            result_province = province_model.predict(license_img)[0]
            # print('---predict: ',result_province,'----')
            inx = np.argmax(result_province)
            province_result = province_name[inx]
            province_prob = result_province[inx]

            # print('ocr time: ', time.time()-start_time_alpr)
            # print("Result : ",result)
            # out = [x.encode('ascii') for x in result]
            # print(type(result[0]))
            # print(out)
            # data = {'text':'text'}\

            vehicle_detail['plate'] = {
                            'result':result,
                            'province_result': province_result,
                            'province_prob': province_prob,
                            'ptspx':, 
                            }

        vehicle_detail['car'] = {
                'car_info':car_info,
                'car_prob':car_prob,
                }
        
        vehicle.append(vehicle_detail)

    data = {
        # 'idx', index,
        'frame':Iorig,
        # 'plate':(Ilp*255.).astype('uint8'),
        # 'plate':LlpImgs,
        'frame_id':frame_id,
        # 'result':result,
        # 'meta':data['meta'],
        # 'test': ptsh,
        # 'car': im2single(Ivehicle),
        'vehicles': vehicle,
    }
    zmq_send.send_pyobj(data)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Elapsed time = %.3f s' % elapsed_time)
    # print('FPS = ', 1.0 / elapsed_time)
    start_time = end_time

import sys, os
import keras
import cv2
import traceback

from src.alpr.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.alpr.utils 					import im2single, nms
from src.alpr.keras_utils 			import load_model, detect_lp
from src.alpr.label 					import Shape, writeShapes, dknet_label_conversion
from time import sleep
import datetime
import numpy as np
import zmq
import base64
import io
from imageio import imread
import tensorflow as tf
import time
import random
from src.lprnet.utils import TextImageGenerator,encode_label,sparse_tuple_from,decode_sparse_tensor,\
                    decode_a_seq,small_basic_block,conv,report_accuracy,\
                    test_report,do_batch,get_test_model,predict_oneimg
import socketio
import pickle
# import matplotlib.pyplot as plt
# def adjust_pts(pts,lroi):
# 	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


'''lpr detection Load model'''
lp_threshold = .5
wpod_net_path = './models/alpr/v2_backup.h5'
wpod_net = load_model(wpod_net_path)


'''ocr lprnet'''
BATCH_SIZE = 1
TRAIN_SIZE = 1000
BATCHES = TRAIN_SIZE//BATCH_SIZE
test_num = 1

ti = 'train_lprnet/lpr_label1.csv'         
vi = 'train_lprnet/lpr_label1.csv'         
img_size = [94, 24]
tl = None
vl = None
num_channels = 3
label_len = 7
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

#global_step = tf.Variable(0, trainable=False)
#print(num_channels, label_len, BATCH_SIZE, img_size)
#logits, inputs, seq_len = get_test_model(num_channels, label_len,BATCH_SIZE, img_size,NUM_CHARS)
#logits = tf.transpose(logits, (1, 0, 2))
#decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)
#init = tf.global_variables_initializer()
#config = tf.ConfigProto(device_count = {'GPU': 1, 'CPU': 16}) 
#config.gpu_options.allow_growth = True

alpr_graph = tf.Graph()
session2 = tf.Session(graph=alpr_graph)
with alpr_graph.as_default():
#with session2.as_default():
    init = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    #with alpr_graph.as_default():
    with session2.as_default():
        tf.global_variables_initializer().run()
        #saver = tf.train.Saver(tf.global_variables())
        saver.restore(session2, 'models/lprnet/LPRtf3.ckpt-300000')

#context = zmq.Context()
#print("Connecting to server...")
#socket = context.socket(zmq.SUB)
#socket.connect("tcp://zmq:6666")
#socket.setsockopt(zmq.SUBSCRIBE, "".encode('utf-8'))

sio = socketio.Client()
sio.connect('http://streamer_detector:5562')
print('Connected')

Ilp = None
#while(True):
@sio.on('raw')
def recive(data):
    #message = (sio.recv().decode('utf-8'))[3:-1]
    ''' Lpr detection and frontalizarion'''
    print ('Searching for license plates using WPOD-NET')
    start = datetime.datetime.now()
    #img = imread(io.BytesIO(base64.b64decode(message)))
    img = pickle.loads(data['frame'], fix_imports=True, encoding="bytes")
    Ivehicle = cv2.imdecode(img, cv2.IMREAD_COLOR)
    #Ivehicle = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    print( "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)
    if len(LlpImgs):
        Ilp = LlpImgs[0]
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
        s = Shape(Llp[0].pts)
        print(type(Ilp))
        ''' Lpr ocr '''
        #frame = cv2.resize(Llp*255.0, (img_size[0], img_size[1]), interpolation=cv2.INTER_CUBIC)
        #predict = predict_oneimg(frame,inputs,seq_len,session,decoded,CHARS)
        #print ('Performing OCR...')
        #print( '\tScanning\t')
        #print(predict)
        #cv2.imshow(frame)
        #cv2.waitkey(1)
    stop = datetime.datetime.now()
    print(stop-start)
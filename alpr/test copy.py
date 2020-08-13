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
        saver.restore(sess, 'models/lprnet/LPRtf3.ckpt-300000')

print(K.get_session() == wpod_sess)


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

    for index,obj in enumerate(data['meta']):
        name = obj['cls']       
        WH = np.array(Iorig.shape[1::-1],dtype=float)
        # print((obj['x'],obj['y'],obj['w'],obj['h']))
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

        Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

        print( "\t\t\tNo. LP: %d" % len(LlpImgs))
        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)
            # print("S : ", Llp[0].pts)
            # print((Ilp*255.).astype('uint8'))

            Ilp = cv2.resize(Ilp,(94,24))
            result = predict_oneimg(Ilp*255.,inputs,seq_len,lprnet_sess,decoded,CHARS)
            # print("Result : ",result)
            # out = [x.encode('ascii') for x in result]
            # print(type(result[0]))
            # print(out)
            # data = {'text':'text'}
            data = {
                'frame':Iorig,
                'plate':(Ilp*255.).astype('uint8'),
                'frame_id':frame_id,
                'result':result,
                'meta':data['meta'],
                'test': "tttt"
            }
            zmq_send.send_pyobj(data)

# elapsed_time = time.time() - start_time
# print('Elapsed time = %.3f s' % elapsed_time)
# print('FPS = %.1f' % (len(img_list) / elapsed_time))
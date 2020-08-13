import numpy as np
import os
import tensorflow as tf
import time
import cv2
import os
import random
import pandas as pd
def encode_label(CHARS_DICT,s):
    label = np.zeros([len(s)])
    print(s)
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label

#读取图片和label,产生batch
class TextImageGenerator:
    def __init__(self, annotate_file, label_file, batch_size, img_size, num_channels=3, label_len=7,CHARS_DICT=None):
        self._annotate_file = annotate_file
        self._label_file = label_file
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = []
        self.CHARS_DICT=CHARS_DICT
        self.init()

    def init(self):
        self.labels = []
#         fs = os.listdir(self._img_dir)
#         for filename in fs:
#             ext = ['png','jpg']
#             if filename.split('.')[-1].lower() not in ext:
#                 continue
#             self.filenames.append(filename)
#         for filename in self.filenames:
#             #if '\u4e00' <= filename[0]<= '\u9fff':
#             #    label = filename[:7]
#             #else:
#             #    label = dict[filename[:3]] + filename[4:10]
#             label = filename.split('_')[0]
#             if len(label) < self._label_len:
#                 label = label + '_'*(self._label_len-len(label))
        df = pd.read_csv(self._annotate_file)
        data = df['lebel']
        for i in data:
            label = encode_label(self.CHARS_DICT,i)
            self.labels.append(label)
            self._num_examples += 1
        self.labels = np.float32(self.labels)
        imgs = df['img_path']
        self.filenames = imgs
        

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]
            
        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end > self._num_examples:
            self._next_index = 0
            start = self._next_index
            end = self._next_index + batch_size
            self._num_epoches += 1
        else:
            self._next_index = end
        images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        # labels = np.zeros([batch_size, self._label_len])

        for j, i in enumerate(range(start, end)):
            # print(j,i)
            fname = self._filenames[i]
#             print(fname)
#             img = cv2.imread(os.path.join(self._img_dir, fname))
            img = cv2.imread(fname)
            img = cv2.resize(img, (self._img_w, self._img_h), interpolation=cv2.INTER_CUBIC)
            images[j, ...] = img
        images = np.transpose(images, axes=[0, 2, 1, 3])
        labels = self._labels[start:end, ...]
        targets = [np.asarray(i) for i in labels]
        sparse_labels = sparse_tuple_from(targets)
        # input_length = np.zeros([batch_size, 1])

        seq_len = np.ones(self._batch_size) * 24
        return images, sparse_labels, seq_len

def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def decode_sparse_tensor(sparse_tensor,CHARS):
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    result = []
    for index in decoded_indexes:
        result.append(decode_a_seq(index, sparse_tensor,CHARS))
    return result


def decode_a_seq(indexes, spars_tensor,CHARS):
    decoded = []
    for m in indexes:
        str = CHARS[spars_tensor[1][m]]
        decoded.append(str)
    return decoded

def small_basic_block(x,im,om):
    x = conv(x,im,int(om/4),ksize=[1,1])
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),int(om/4),ksize=[3,1],pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),int(om/4),ksize=[1,3],pad='SAME')
    x = tf.nn.relu(x)
    x = conv(x,int(om/4),om,ksize=[1,1])
    return x

def conv(x,im,om,ksize,stride=[1,1,1,1],pad = 'SAME'):
    conv_weights = tf.Variable(
        tf.truncated_normal([ksize[0], ksize[1], im, om],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=None, dtype=tf.float32))
    conv_biases = tf.Variable(tf.zeros([om], dtype=tf.float32))
    out = tf.nn.conv2d(x,
                        conv_weights,
                        strides=stride,
                        padding=pad)
    relu = tf.nn.bias_add(out, conv_biases)
    return relu

def get_train_model(num_channels, label_len, b, img_size,NUM_CHARS):
    inputs = tf.placeholder(
        tf.float32,
        shape=(b, img_size[0], img_size[1], num_channels))

    # 定义ctc_loss需要的稀疏矩阵
    targets = tf.sparse_placeholder(tf.int32)

    # 1维向量 序列长度 [batch_size,]
    seq_len = tf.placeholder(tf.int32, [None])
    x = inputs

    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    x = small_basic_block(x,64,64)
    x2=x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')
    x = small_basic_block(x, 64,256)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x)

    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    x = conv(x, 256, 256, ksize=[4, 1])
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)

    #x = tf.reduce_mean(x,axis = 2)
    #x1 = conv(inputs,num_channels,num_channels,ksize = (5,1))


    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)

    # x1 = tf.image.resize_images(x1, size = [18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)

    #x2 = tf.image.resize_images(x2, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    #x3 = tf.image.resize_images(x3, size=[18, 16], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


    #x1 = tf.nn.relu(x1)

    x = tf.concat([x,x1,x2,x3],3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))
    logits = tf.reduce_mean(x,axis=2)
    # x_shape = x.get_shape().as_list()
    # outputs = tf.reshape(x, [-1,x_shape[2]*x_shape[3]])
    # W1 = tf.Variable(tf.truncated_normal([x_shape[2]*x_shape[3],
    #                                      150],
    #                                     stddev=0.1))
    # b1 = tf.Variable(tf.constant(0., shape=[150]))
    # # [batch_size*max_timesteps,num_classes]
    # x = tf.matmul(outputs, W1) + b1
    # x= tf.layers.dropout(x)
    # x = tf.nn.relu(x)
    # W2 = tf.Variable(tf.truncated_normal([150,
    #                                      NUM_CHARS+1],
    #                                     stddev=0.1))
    # b2 = tf.Variable(tf.constant(0., shape=[NUM_CHARS+1]))
    # x = tf.matmul(x, W2) + b2
    # x = tf.layers.dropout(x)
    # # [batch_size,max_timesteps,num_classes]
    # logits = tf.reshape(x, [b, -1, NUM_CHARS+1])

    return logits, inputs, targets, seq_len

def report_accuracy(decoded_list, test_targets,CHARS):
    original_list = decode_sparse_tensor(test_targets,CHARS)
    detected_list = decode_sparse_tensor(decoded_list,CHARS)
    true_numer = 0

    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
        detect_number = detected_list[idx]
        hit = (number == detect_number)
        print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
        if hit:
            true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / len(original_list))

def do_report(val_gen,num,inputs,targets,seq_len,session,decoded,CHARS):
    for i in range(num):
        test_inputs, test_targets, test_seq_len = val_gen.next_batch()
        test_feed = {inputs: test_inputs,
                    targets: test_targets,
                    seq_len: test_seq_len}
        st =time.time()
        dd= session.run(decoded[0], test_feed)
        tim = time.time() -st
        print('time:%s'%tim)
        report_accuracy(dd, test_targets,CHARS)

def test_report(testi,files):
    true_numer = 0
    num = files//BATCH_SIZE

    for i in range(num):
        test_inputs, test_targets, test_seq_len = val_gen.next_batch()
        test_feed = {inputs: test_inputs,
                    targets: test_targets,
                    seq_len: test_seq_len}
        dd = session.run([decoded[0]], test_feed)
        original_list = decode_sparse_tensor(test_targets,CHARS)
        detected_list = decode_sparse_tensor(dd,CHARS)
        for idx, number in enumerate(original_list):
            detect_number = detected_list[idx]
            hit = (number == detect_number)
            if hit:
                true_numer = true_numer + 1
    print("Test Accuracy:", true_numer * 1.0 / files)


def do_batch(train_gen,val_gen,inputs,targets,seq_len,loss,logits,cost,global_step,optimizer,saver,session,REPORT_STEPS,test_num,decoded,CHARS):
    train_inputs, train_targets, train_seq_len = train_gen.next_batch()

    feed = {inputs: train_inputs, targets: train_targets, seq_len: train_seq_len}

    b_loss, b_targets, b_logits, b_seq_len, b_cost, steps, _ = session.run(
        [loss, targets, logits, seq_len, cost, global_step, optimizer], feed)
    # print(steps," ",REPORT_STEPS)
    #print(b_cost, steps)
    if steps > 0 and steps % REPORT_STEPS == 0:
        do_report(val_gen,test_num,inputs,targets,seq_len,session,decoded,CHARS)
        saver.save(session, "./model_global/LPRtf3.ckpt", global_step=steps)
    return b_cost, steps

def get_test_model(num_channels, label_len, b, img_size,NUM_CHARS):
    inputs = tf.placeholder(
        tf.float32,
        shape=(b, img_size[0], img_size[1], num_channels))

    seq_len = tf.placeholder(tf.int32, [None])
    x = inputs

    x = conv(x,num_channels,64,ksize=[3,3])
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 1, 1, 1],
                          padding='SAME')
    x = small_basic_block(x,64,64)
    x2=x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)

    x = tf.nn.max_pool(x,
                          ksize=[1, 3, 3, 1],
                          strides=[1, 2, 1, 1],
                          padding='SAME')
    x = small_basic_block(x, 64,256)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)
    x = small_basic_block(x, 256, 256)
    x3 = x
    x = tf.layers.batch_normalization(x)

    x = tf.nn.relu(x)
    x = tf.nn.max_pool(x,
                       ksize=[1, 3, 3, 1],
                       strides=[1, 2, 1, 1],
                       padding='SAME')
    x = tf.layers.dropout(x)

    x = conv(x, 256, 256, ksize=[4, 1])
    x = tf.layers.dropout(x)
    x = tf.layers.batch_normalization(x)
    x = tf.nn.relu(x)


    x = conv(x,256,NUM_CHARS+1,ksize=[1,13],pad='SAME')
    x = tf.nn.relu(x)
    cx = tf.reduce_mean(tf.square(x))
    x = tf.div(x,cx)


    x1 = tf.nn.avg_pool(inputs,
                       ksize=[1, 4, 1, 1],
                       strides=[1, 4, 1, 1],
                       padding='SAME')
    cx1 = tf.reduce_mean(tf.square(x1))
    x1 = tf.div(x1, cx1)


    x2 = tf.nn.avg_pool(x2,
                        ksize=[1, 4, 1, 1],
                        strides=[1, 4, 1, 1],
                        padding='SAME')
    cx2 = tf.reduce_mean(tf.square(x2))
    x2 = tf.div(x2, cx2)


    x3 = tf.nn.avg_pool(x3,
                        ksize=[1, 2, 1, 1],
                        strides=[1, 2, 1, 1],
                        padding='SAME')
    cx3 = tf.reduce_mean(tf.square(x3))
    x3 = tf.div(x3, cx3)

    x = tf.concat([x,x1,x2,x3],3)
    x = conv(x, x.get_shape().as_list()[3], NUM_CHARS + 1, ksize=(1, 1))
    logits = tf.reduce_mean(x,axis=2)
    return logits, inputs, seq_len

def predict_oneimg(img,inputs,seq_len,session,decoded,CHARS):
    img = np.transpose(img, axes=[1, 0, 2])
    
    test_feed = {inputs: img[np.newaxis, :],
                 seq_len: (24,)}
    st =time.time()
    dd= session.run(decoded[0], test_feed)
    tim = time.time() -st
    print('time:%s'%tim)
    detect_numbers = decode_sparse_tensor(dd, CHARS)
    for detect_number in detect_numbers:
        # print(detect_number, "(", len(detect_number), ")")
        text = u' '.join(detect_number)
        return text
#! /usr/bin/env python

# Version withow buckets
# Version 04: Add a restart mechanism
# Execute using py3_tf1 environment


from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import cv2
import os
import time
import h5py
import editdistance
import pickle


# Path to the data
data_path = './data/'





# data_utils.py
#=============================
def decoder_dict_database(iam_patches):
    '''
    Load decoder dictionary for IAM database
    '''
    #Recover decoding dictionary
    with h5py.File(iam_patches, "r") as hdf5_f:
        keys_dict = np.copy(hdf5_f["target_dict_keys"])
        values_dict = np.copy(hdf5_f["target_dict_values"])
    decoder_dict={}
    for i, key in enumerate(keys_dict):
        decoder_dict[key] = values_dict[i].decode('UTF-8')

    # Complete decoding dictionary
    num_characters = len(decoder_dict)
    PAD_ID = num_characters
    GO_ID = num_characters+1
    EOL_ID = num_characters+2
    decoder_dict[PAD_ID]='-PAD'
    decoder_dict[GO_ID]='GO'
    decoder_dict[EOL_ID]='-EOL'

    return PAD_ID, GO_ID, EOL_ID, decoder_dict


def decode_text(text_array, decoder_dict):
    '''
    Decode the target from numbers to words
    '''
    text = ''
    eol_code = len(decoder_dict)-1
    ind_eol = False
    for c in text_array:
        if ind_eol==False:
            text += decoder_dict[c]
        if c==eol_code:
            ind_eol=True
    return text
# print(decode_text(np.array([44, 53, 67, 57, 71, 53, 81, 81], dtype=np.uint8))) #Sandra




# Image augmentation code
# ========================
def move_img(img):
    pixels_move = 1 + int(random.random()*10)
    img2 = np.ones_like(img)*0
    img2[:,pixels_move:] = img[:,:-pixels_move]
    return img2

def resize_down(img):
    factor = 0.95 - random.random()/4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img,None,fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = np.ones_like(img)*0
    img2[(h_ini-h_fin)//2:-(h_ini-h_fin)//2, :w_fin] = img1
    return img2

def resize_up(img):
    factor = 1 + random.random()/4.
    h_ini, w_ini = img.shape
    img1 = cv2.resize(img, None, fx=factor, fy=factor, interpolation = cv2.INTER_CUBIC)
    h_fin, w_fin = img1.shape
    img2 = img1[h_fin-h_ini:, :w_ini]
    return img2

def img_augmented(img):
    # Move left
    img = move_img(img)

    # Skew
    if random.random() < 0.8 :
        shape_ini = img.shape
        angle = (random.random()-0.5)/3.
        M = np.float32([[1, -angle, 0.5*img.shape[0]*angle], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)

    #Resize
    if random.random() < 0.4:
        img = resize_down(img)
    elif random.random() < 0.4:
        img = resize_up(img)

    #Erode - dilate
    if random.random() < 0.3:
        img = cv2.erode(img, np.ones(2, np.uint8), iterations=1)
    elif random.random() < 0.3:
        img = cv2.dilate(img, np.ones(2, np.uint8), iterations=1)

    return img




#Generate slides of the image
def generate_images(img_batch, img_len_batch, x_slide_size = 28, slides_stride = 2, normalize=True, augmentation=True):
    #Normalize batch
    if normalize:
        img_batch_normalized =  1 - (img_batch/255.)
    else:
        img_batch_normalized = img_batch

    images_batch = []
    slides_len_batch = []
    # Convert img_batch in a sequence of frames and calculate slides_len_batch
    for n_img, img in enumerate(img_batch_normalized):
        #Data augmentation
        if augmentation:
            try:
                img = img_augmented(img)
            except:
                print('Error in data augmentation')
        images_batch += [img]
        #Calculate slides_len_batch as the number of slides to get
        max_slides = int((img.shape[1] - x_slide_size)/float(slides_stride))
        num_slides = max(2,min(max_slides, 1 + int((img_len_batch[n_img] - x_slide_size)/float(slides_stride))))
        slides_len_batch += [num_slides]

    return np.array(images_batch), np.array(slides_len_batch)



def generate_target(y_ini, y_len, seq_length=19, num_classes=81+3):

    #Create vars: target, dec_inp and weigth
    batch_size = y_ini.shape[0]
    decoder_inputs = np.zeros([batch_size, seq_length+1, num_classes], dtype=np.float32)
    weights = np.zeros([batch_size, seq_length+1], dtype=np.float32)
    targets = np.zeros([batch_size, seq_length+1], dtype=np.uint16)
    for batch_i in range(batch_size):
        for char_pos in range(seq_length+1):
            if char_pos == 0:
                decoder_inputs[batch_i, char_pos, GO_ID] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = int(y_ini[batch_i, char_pos])
            elif char_pos < y_len[batch_i]:
                decoder_inputs[batch_i, char_pos, int(y_ini[batch_i, char_pos-1])] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = int(y_ini[batch_i, char_pos])
            elif char_pos == y_len[batch_i]:
                decoder_inputs[batch_i, char_pos, int(y_ini[batch_i, char_pos-1])] = 1
                weights[batch_i, char_pos] = weights_dict[char_pos]
                targets[batch_i, char_pos] = EOL_ID
            else:
                decoder_inputs[batch_i, char_pos, PAD_ID] = 1
                weights[batch_i, char_pos] = 0
                targets[batch_i, char_pos] = PAD_ID

    return decoder_inputs, targets, weights


#Extract data to numpy arrays and generate batch over it

def extract_data(hdf5_file, dset_type):
    with h5py.File(hdf5_file, "r") as hdf5_f:
        img    = np.copy(hdf5_f.get('X_'+dset_type))
        img_l  = np.copy(hdf5_f.get('image_length_'+dset_type))
        target   = np.copy(hdf5_f.get('target_'+dset_type))
        target_l = np.copy(hdf5_f.get('target_length_'+dset_type))
    return img, img_l, target, target_l


def batch_generator_epoch(img, img_l, target, target_l, batch_size=256,
                          seq_length=19, slides_stride=5, x_slide_size=28,
                          num_classes=81+3, augmentation=True):
    '''
    Generator to one epoch of data
    '''
    #Randomize batches
    data_size = img.shape[0]
    p_index = np.random.permutation(range(0, data_size))
    batch_list=[]
    for b in range(0, data_size, batch_size):
        batch_list += [list(p_index[b:b+batch_size])]

    # Iterate over each batch
    for batch in batch_list:
        #Extract batch data
        img_b      = img[batch]
        img_l_b    = img_l[batch]
        target_b   = target[batch]
        target_l_b = target_l[batch]

        images_batch, slides_len_batch = generate_images(img_b, img_l_b,
                                        x_slide_size = x_slide_size, slides_stride = slides_stride, augmentation=augmentation)

        decoder_inputs, targets, weights = generate_target(target_b, target_l_b,
                                               seq_length=seq_length, num_classes=num_classes)

        yield  images_batch, slides_len_batch, decoder_inputs, targets, weights, img_b, target




# Evaluation functions
# ==================================================

# Real test dictionary
def get_real_test_dictionary_set(iam_patches):
    with h5py.File(iam_patches, "r") as hdf5_f:
        target_tst = np.copy(hdf5_f["target_tst"])
        target_length_tst = np.copy(hdf5_f["target_length_tst"])
    target_tst_set = set([decode_text(target_tst[i][:target_length_tst[i]], decoder_dict) for i in range(len(target_tst))])
    print('Length of test dictionary:', len(target_tst_set))
    return target_tst_set
#real_test_dict_set = get_real_test_dictionary_set(data_path + 'IAM_normalized_words_48px_patches.h5')


def calculate_wer_cer(real, predict, predict_ori, print_results=False):
    WER = 0
    CER = 0
    n_chars = 0
    errors = 0
    for i in range(len(real)):
        CER += editdistance.eval(real[i], predict[i])
        n_chars += len(real[i])
        if real[i] != predict[i]:
            WER += 1
            errors += 1
            if print_results:
                if errors<100:
                    print('i:',i,'real:', real[i], ' - pred original:',predict_ori[i], ' - pred:',predict[i])

    return float(WER)/len(real), float(CER)/n_chars


def evaluate_corpus(real, predict, dictionary, print_results=False):
    #for each word in the predict list, find the closed word in the test_set.
    predict_lng_model=[]
    for w in predict:
        min_distance=1000
        closed_word=''
        for wt in dictionary:
            if editdistance.eval(w, wt) < min_distance:
                min_distance = editdistance.eval(w, wt)
                closed_word = wt
        predict_lng_model += [closed_word]

    if print_results:
        print(len(predict_lng_model))

    return calculate_wer_cer(real, predict_lng_model, predict, print_results=print_results)









# Parameters
# ==================================================

# Misc Parameters
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices (default False)")
tf.flags.DEFINE_string("cuda_device", '0', "GPU device (default 0)")

# Data loading params
#tf.flags.DEFINE_integer("size", 10, "size (default: 10)")


tf.flags.DEFINE_string("train_data", 'rimes_words_48_192.hdf5',
                       "Train data(default rimes_words_48_192.hdf5")
tf.flags.DEFINE_integer("x_shape", 192, "x_shape (default: 190)")
tf.flags.DEFINE_integer("y_shape", 48, "y_shape (default: 48)")
tf.flags.DEFINE_integer("x_slide_size", 28, "x_slide_size (default: 28)")
tf.flags.DEFINE_integer("slides_stride", 2, "slides_stride (default: 2)")
tf.flags.DEFINE_integer("seq_decoder_len", 19, "max_length of a word (default: 16)")

# Char model parameters
tf.flags.DEFINE_string("character_model", data_path + 'char_models/lenet_x28_IAM',
                       "Char model(default /home/jorge/data/tesis/handwriting/p07_seq2seq/char_models/lenet_x28_IAM)")
tf.flags.DEFINE_boolean("train_pretrained_char_model", False, "Use a pretrained char model (default False)")
tf.flags.DEFINE_boolean("use_pretrained_char_model", True, "Use a pretrained char model (default True)")
tf.flags.DEFINE_boolean("trainable_char_model", True, "Make the char model part trainable (default True)")
tf.flags.DEFINE_integer("dense_size_char_model", 1024, "dense size of the char model (default: 1024)")
tf.flags.DEFINE_boolean("use_lenet", True, "Use Lenet(default) achitecture vs VGG architecture (--nouse_lenet)")




# Model Hyperparameters
# RNN parameters
tf.flags.DEFINE_boolean("bidirectional", True, "Use a bidirectional model (default True)")
tf.flags.DEFINE_integer("num_layers", 1, "num_layers (default: 1)")
tf.flags.DEFINE_integer("dim_lstm", 256, "dim_lstm (default: 256)")
tf.flags.DEFINE_integer("num_heads", 1, "num_heads (default: 1)")
tf.flags.DEFINE_float("keep_prob", 0.5, "keep_prob (default: 0.5)")
tf.flags.DEFINE_float("lambda_l2_reg", 0, "lambda_l2_reg (default: 0)")

# Training parameters
tf.flags.DEFINE_string("experiment", 'exp01',"Experiment name dir (default exp01)")
tf.flags.DEFINE_string("load_model_name", '',
                       "If continue training, name of the model to load (default <BLANK> no continue training)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_float("pct_lr_char_model", 0.1, "Percent of learning rate applied to the char model part (default: 0.1)")
tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 256)")
tf.flags.DEFINE_integer("exponential_decay_step", 100, "exponential_decay_step (defaults 100)")
tf.flags.DEFINE_float("exponential_decay_rate", 0.95, "exponential_decay_rate (default 0.95)")
tf.flags.DEFINE_integer("min_steps", 10, "min_steps (defaults 10 - min 10)")
tf.flags.DEFINE_integer("max_steps", 1000, "max_steps (defaults 1000 - min 1)")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")




# Data generator parameters
#==========================

#weights_dictionary
weights_dict={0:10, 1:5, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1,
             10:1, 11:1, 12:1, 13:1, 14:1, 15:1, 16:1, 17:1, 18:1, 19:1, 20:1}

PAD_ID, GO_ID, EOL_ID, decoder_dict = decoder_dict_database(data_path + FLAGS.train_data)
print('decoder_dict: ', decoder_dict)
# The last 3 characters of the dict are the special characters PAD, GO and END.
num_classes = len(decoder_dict)
num_characters = num_classes - 3



img_trn, img_l_trn, target_trn, target_l_trn = extract_data(data_path + FLAGS.train_data, 'trn')
img_val, img_l_val, target_val, target_l_val = extract_data(data_path + FLAGS.train_data, 'val')
img_tst, img_l_tst, target_tst, target_l_tst = extract_data(data_path + FLAGS.train_data, 'tst')




#Limit GPU cards
#==========================
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.cuda_device
#gpu_options = tf.GPUOptions(allow_growth = True)
gpu_options = tf.GPUOptions(allow_growth = False)




# Char model functions
#==========================
def train_tests_character_data(new_size = 48):

    import matplotlib.pyplot as plt
    import cv2
    from os.path import isdir, isfile, join
    from os import listdir

    chars_path = '/home/jorge/data/tesis/handwriting/databases/handwriting_characters/curated/'

    # Recover seq2seq target dictionary
    _, _, _, decoder_dict = decoder_dict_database(data_path + FLAGS.train_data)

    # Inverse dict
    encoder_dict = {v: k for k, v in decoder_dict.items()}

    X_list = []
    y_list = []
    for c in list(decoder_dict.values())[:num_characters]:
        mypath = os.path.join(chars_path, str(ord(c)))
        if isdir(mypath):
            for f_name in [f for f in listdir(mypath) if isfile(join(mypath, f))]:
                new_img = cv2.resize(plt.imread(os.path.join(mypath, f_name)), (new_size, new_size))
                if len(new_img.shape)==3:
                    print(os.path.join(mypath, f_name))
                X_list += [new_img]
                y_list += [encoder_dict[c]]

    # shuffle and separate train test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_list), np.array(y_list), test_size=0.10, random_state=42)
    print('X/y char train shape:', X_train.shape, y_train.shape)
    print('X/y char test  shape:', X_test.shape, y_test.shape)

    # Print y_test distribution
    y_count = np.bincount(y_test)
    ii = np.nonzero(y_count)[0]
    print('y_test values distribution')
    print(zip(ii,y_count[ii]) )

    return X_train, X_test, y_train, y_test, num_characters


# Create batch generator for character database
def char_batch_generator(X, y, batch_size=256, y_shape=48, x_shape=28, num_characters = 78):
    batch_randomized = np.random.permutation(range(0, X.shape[0]-batch_size, batch_size))
    for n_batch, batch in enumerate(batch_randomized):
        X_batch = np.zeros([batch_size, y_shape, x_shape])
        y_batch = np.zeros([batch_size, num_characters], dtype=np.uint8)
        for i, index in enumerate(range(batch, batch+batch_size)):
            # Select a patch of y_shape x x_shape
            if y_shape<=x_shape:
                pos_ini_x = 0
            else:
                pos_ini_x = np.random.randint(0, y_shape-x_shape) # [0,48-28]
            X_batch[i] = X[index, :, pos_ini_x:pos_ini_x+x_shape]
            y_batch[i, y[index]] = 1
        yield np.array(X_batch, dtype=np.float32), y_batch
# Test
'''
gen = char_batch_generator(X_train, y_train, batch_size=2)
X_b, y_b = gen.next()
print(X_b.shape)
print(y_b)
print(np.max(X_b))
plt.imshow(X_b[0])
'''



def train_char_model():

    print('Training the char model')
    X_train, X_test, y_train, y_test, num_characters = train_tests_character_data(new_size = FLAGS.y_shape)


    graph = tf.Graph()
    with graph.as_default():

        #Create the net
        # Inputs
        x = tf.placeholder(tf.float32, shape=[None, FLAGS.y_shape, FLAGS.x_slide_size])
        y = tf.placeholder(tf.float32, shape=[None, num_characters])

        #-----------   MODEL  ---------------
        #Reshape input data to the original image shape
        x_image = tf.reshape(x, [-1, FLAGS.y_shape, FLAGS.x_slide_size, 1])

        # First convolution
        W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1))
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[20]))
        h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('Conv - pool 1: ', h_pool1)

        #Second convolution
        W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 20, 50], stddev=0.1))
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[50]))
        h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
        h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        print('Conv - pool 2: ', h_pool2)


        #First dense layer
        h_pool2_flat = tf.contrib.layers.flatten(h_pool2)
        dim_pool = h_pool2_flat.get_shape()
        print('Dim pool: ', dim_pool[1].value)

        W_d1 = tf.Variable(tf.truncated_normal([dim_pool[1].value, FLAGS.dense_size_char_model], stddev=0.1))
        b_d1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.dense_size_char_model]))
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_d1) + b_d1)
        print('Dense 1: ', h_fc1)

        #Dropout
        dropout_keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, dropout_keep_prob)

        #Second dense layer
        W_d2 = tf.Variable(tf.truncated_normal([FLAGS.dense_size_char_model, num_characters], stddev=0.1))
        b_d2 = tf.Variable(tf.constant(0.1, shape=[num_characters]))
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_d2) + b_d2)
        print('Dense 2: ', h_fc2)

        #Prediction
        y_pred = tf.nn.softmax(h_fc2)

        # Loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=h_fc2, name='cross_entropy')

        #Optimizer
        train_step = tf.train.AdamOptimizer(1e-3, name='adam_char').minimize(cross_entropy)

        #Accuracy
        correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #Inicialization.
        sess.run(tf.global_variables_initializer())

        for epoch in range(100):
            train_accuracy = []
            test_accuracy  = []

            # Train step
            batch_generator = char_batch_generator(X_train, y_train, batch_size=256, y_shape=FLAGS.y_shape,
                                                   x_shape=FLAGS.x_slide_size, num_characters = num_characters)

            for i, batch in enumerate(batch_generator):
                train_step.run(feed_dict={x: batch[0], y: batch[1], dropout_keep_prob: 0.5})
                train_accuracy += [accuracy.eval(feed_dict={x: batch[0], y: batch[1], dropout_keep_prob: 1})]

            # Test step
            batch_generator = char_batch_generator(X_test, y_test, batch_size=256, y_shape=FLAGS.y_shape,
                                                   x_shape=FLAGS.x_slide_size, num_characters = num_characters,
                                                   augmentation=False)
            for i, batch in enumerate(batch_generator):
                test_accuracy += [accuracy.eval(feed_dict={x: batch[0], y: batch[1],
                                                                 dropout_keep_prob: 1})]

            print("Char model. Step %d, train accuracy %g, test accuracy %g"%(epoch, np.mean(train_accuracy), np.mean(test_accuracy)))


        # Create a saver and save weigths.
        char_saver = tf.train.Saver(max_to_keep=0)

        # Add to collection
        tf.add_to_collection('W_conv1', W_conv1)
        tf.add_to_collection('b_conv1', b_conv1)
        tf.add_to_collection('W_conv2', W_conv2)
        tf.add_to_collection('b_conv2', b_conv2)
        tf.add_to_collection('W_d1', W_d1)
        tf.add_to_collection('b_d1', b_d1)
        tf.add_to_collection('W_d2', W_d2)
        tf.add_to_collection('b_d2', b_d2)

        model_saved_file = char_saver.save(sess, FLAGS.character_model,)
        print('Char model saved in :', model_saved_file)

    # Reset the graph to the next experiments
    tf.reset_default_graph()



def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/'   + name, mean)
        tf.summary.scalar('sttdev/' + name, tf.sqrt(tf.reduce_mean(tf.square(var - mean))))
        tf.summary.scalar('max/'    + name, tf.reduce_max(var))
        tf.summary.scalar('min/'    + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


def lenet_over_seq_pretrained(img_seq, dropout_keep_prob):
    print('-- Lenet pretrained model. Trainable=', FLAGS.trainable_char_model)
    #First convolution
    W_conv_1 = tf.Variable(np_W_conv1, trainable=FLAGS.trainable_char_model)
    b_conv_1 = tf.Variable(np_b_conv1, trainable=FLAGS.trainable_char_model)
    conv1_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1) for x_in in img_seq]
    h_pool1 = [tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv1 in conv1_out]
    variable_summaries(W_conv_1, 'W_conv_1')


    #Second convolution
    W_conv_2 = tf.Variable(np_W_conv2, trainable=FLAGS.trainable_char_model)
    b_conv_2 = tf.Variable(np_b_conv2, trainable=FLAGS.trainable_char_model)
    conv2_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2) for x_in in h_pool1]
    h_pool2 = [tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv2 in conv2_out]
    variable_summaries(W_conv_2, 'W_conv_2')

    #First dense layer
    h_pool2_flat = [tf.contrib.layers.flatten(hp) for hp in h_pool2]

    W_dense_1 = tf.Variable(np_W_d1, trainable=FLAGS.trainable_char_model)
    b_dense_1 = tf.Variable(np_b_d1, trainable=FLAGS.trainable_char_model)
    dense_output_1 = [tf.nn.relu(tf.matmul(x_in, W_dense_1) + b_dense_1) for x_in in h_pool2_flat]

    #Dropout over
    #h_fc1_drop = [tf.nn.dropout(h_fc1, dropout_keep_prob) for h_fc1 in dense_output_1]

    #Second dense layer
    #W_dense_2 = tf.Variable(np_W_d2, trainable=FLAGS.trainable_char_model)
    #b_dense_2 = tf.Variable(np_b_d2, trainable=FLAGS.trainable_char_model)
    #dense_output_2 = [tf.nn.relu(tf.matmul(x_in, W_dense_2) + b_dense_2) for x_in in h_fc1_drop]

    return dense_output_1


def lenet_over_seq(img_seq, dropout_keep_prob):
    print('-- Lenet  model.')
    #First convolution
    W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 20], stddev=0.1))
    b_conv_1 = tf.Variable(tf.constant(0.1, shape=[20]), name='bias_c1')
    conv1_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1) for x_in in img_seq]
    h_pool1 = [tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv1 in conv1_out]
    variable_summaries(W_conv_1, 'W_conv_1')

    #Second convolution
    W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 20, 50], stddev=0.1))
    b_conv_2 = tf.Variable(tf.constant(0.1, shape=[50]), name='bias_c2')
    conv2_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2) for x_in in h_pool1]
    h_pool2 = [tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv2 in conv2_out]
    variable_summaries(W_conv_2, 'W_conv_2')

    #First dense layer
    h_pool2_flat = [tf.contrib.layers.flatten(hp) for hp in h_pool2]
    dim_pool = h_pool2_flat[0].get_shape()

    W_dense_1 = tf.Variable(tf.truncated_normal([dim_pool[1].value, FLAGS.dense_size_char_model], stddev=0.1))
    b_dense_1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.dense_size_char_model]), name='bias_d1')
    dense_output_1 = [tf.nn.relu(tf.matmul(x_in, W_dense_1) + b_dense_1) for x_in in h_pool2_flat]

    #Dropout over
    #h_fc1_drop = [tf.nn.dropout(h_fc1, dropout_keep_prob) for h_fc1 in dense_output_1]

    #Second dense layer
    #W_dense_2 = tf.Variable(tf.truncated_normal([FLAGS.dense_size_char_model, num_characters], stddev=0.1))
    #b_dense_2 = tf.Variable(tf.constant(0.1, shape=[num_characters]))
    #dense_output_2 = [tf.nn.relu(tf.matmul(x_in, W_dense_2) + b_dense_2) for x_in in h_fc1_drop]

    return dense_output_1


def VGG_over_seq(img_seq, dropout_keep_prob):
    print('-- Lenet  model.')
    #First convolution
    W_conv_11 = tf.Variable(tf.truncated_normal([3, 3, 1, 32], stddev=0.1))
    b_conv_11 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias_c11')
    conv11_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_11, strides=[1, 1, 1, 1], padding='SAME') + b_conv_11) for x_in in img_seq]

    W_conv_12 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1))
    b_conv_12 = tf.Variable(tf.constant(0.1, shape=[32]), name='bias_c12')
    conv12_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_12, strides=[1, 1, 1, 1], padding='SAME') + b_conv_12) for x_in in conv11_out]

    h_pool1 = [tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv1 in conv12_out]
    variable_summaries(W_conv_11, 'W_conv_12')

    #Second convolution
    W_conv_21 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1))
    b_conv_21 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_c21')
    conv21_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_21, strides=[1, 1, 1, 1], padding='SAME') + b_conv_21) for x_in in h_pool1]

    W_conv_22 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev=0.1))
    b_conv_22 = tf.Variable(tf.constant(0.1, shape=[64]), name='bias_c22')
    conv22_out = [tf.nn.relu(tf.nn.conv2d(x_in, W_conv_22, strides=[1, 1, 1, 1], padding='SAME') + b_conv_22) for x_in in conv21_out]

    h_pool2 = [tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') for h_conv2 in conv22_out]
    variable_summaries(W_conv_21, 'W_conv_22')

    #First dense layer
    h_pool2_flat = [tf.contrib.layers.flatten(hp) for hp in h_pool2]
    dim_pool = h_pool2_flat[0].get_shape()

    W_dense_1 = tf.Variable(tf.truncated_normal([dim_pool[1].value, FLAGS.dense_size_char_model], stddev=0.1))
    b_dense_1 = tf.Variable(tf.constant(0.1, shape=[FLAGS.dense_size_char_model]), name='bias_d1')
    dense_output_1 = [tf.nn.relu(tf.matmul(x_in, W_dense_1) + b_dense_1) for x_in in h_pool2_flat]

    #Dropout over
    #h_fc1_drop = [tf.nn.dropout(h_fc1, dropout_keep_prob) for h_fc1 in dense_output_1]

    #Second dense layer
    #W_dense_2 = tf.Variable(tf.truncated_normal([FLAGS.dense_size_char_model, num_characters], stddev=0.1))
    #b_dense_2 = tf.Variable(tf.constant(0.1, shape=[num_characters]))
    #dense_output_2 = [tf.nn.relu(tf.matmul(x_in, W_dense_2) + b_dense_2) for x_in in h_fc1_drop]

    return dense_output_1





# Train pretrained char model weigths to numpy arrays
#==========================
if FLAGS.train_pretrained_char_model:
    train_char_model()



# Load pretrained char model weigths to numpy arrays
#==========================
if FLAGS.use_pretrained_char_model:
    graph = tf.Graph()
    with graph.as_default():
        #Load graph
        load_saver = tf.train.import_meta_graph(FLAGS.character_model+'.meta')

    with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        #Load weigths
        load_saver.restore(sess, FLAGS.character_model)
        np_W_conv1 = tf.get_collection('W_conv1')[0].eval()
        np_b_conv1 = tf.get_collection('b_conv1')[0].eval()
        np_W_conv2 = tf.get_collection('W_conv2')[0].eval()
        np_b_conv2 = tf.get_collection('b_conv2')[0].eval()
        np_W_d1 = tf.get_collection('W_d1')[0].eval()
        np_b_d1 = tf.get_collection('b_d1')[0].eval()
        np_W_d2 = tf.get_collection('W_d2')[0].eval()
        np_b_d2 = tf.get_collection('b_d2')[0].eval()
    tf.reset_default_graph()
    print('Pretrained char model loaded!')





### Create model
### ==============================================================

graph = tf.Graph()
with graph.as_default():

    #Placeholders
    with tf.name_scope('inputs') as scope:
        input_image = tf.placeholder(np.float32,[None, FLAGS.y_shape, FLAGS.x_shape], name='input_image')

        input_slides_len = tf.placeholder(tf.int32, shape=(None), name='input_word_len')

        input_word_chars = tf.placeholder(tf.float32, shape=(None, FLAGS.seq_decoder_len+1, num_classes),
                                          name="input_word_chars")
        input_decoder = [tf.reshape(t, [-1, num_classes]) for t in tf.split(input_word_chars, FLAGS.seq_decoder_len+1, axis=1)]

        input_targets = tf.placeholder(tf.int32  , shape=[None, FLAGS.seq_decoder_len+1], name='input_targets')
        input_weights = tf.placeholder(tf.float32, shape=[None, FLAGS.seq_decoder_len+1], name='input_weights')

        weights = [tf.reshape(t, [-1]) for t in tf.split(input_weights, FLAGS.seq_decoder_len+1, axis=1 )]
        targets = [tf.reshape(t, [-1]) for t in tf.split(input_targets, FLAGS.seq_decoder_len+1, axis=1 )]

        keep_prob = tf.placeholder(tf.float32)

    #Transform images to input to the LSTM encoder
    with tf.name_scope('convolutions') as scope:
        # Extract patches of original image
        image_reshape = tf.reshape(input_image, [-1, FLAGS.y_shape, FLAGS.x_shape, 1]) # Add color channel
        patches  = tf.extract_image_patches(image_reshape, [1, FLAGS.y_shape, FLAGS.x_slide_size, 1],
                                        [1, 1, FLAGS.slides_stride, 1], [1, 1, 1, 1], 'VALID' ) #Dim [b, n_patches, y_patch*x_patch]

        seq_encoder_len = patches.get_shape()[2].value
        patches_reshaped = tf.reshape(patches, [-1, seq_encoder_len, FLAGS.y_shape, FLAGS.x_slide_size]) #Dim [b, n_patches, y_patch, x_patch]

        # Generate input convolutions
        input_convolution = [tf.transpose(t, perm=[0, 2, 3, 1]) for t in tf.split(patches_reshaped, seq_encoder_len, axis=1)]
        #print('-- Input_convolution: ', input_convolution) # List of [b, y_patch, x_patch] of size num_patches


        if FLAGS.use_pretrained_char_model:
            input_encoder_list = lenet_over_seq_pretrained(input_convolution, keep_prob)
        else:
            if FLAGS.use_lenet:
                input_encoder_list = lenet_over_seq(input_convolution, keep_prob)
            else:
                input_encoder_list = VGG_over_seq(input_convolution, keep_prob)
        input_encoder = tf.stack(input_encoder_list, axis=1)
        #print('-- input_encoder: ', input_encoder)
        variable_summaries(input_encoder, 'input_encoder')


    ## ENCODER
    with tf.name_scope('encoder') as scope:
        if FLAGS.bidirectional: # Bideirectional model
            with tf.variable_scope('BLSTM1') as scope:
                cell_fw1 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
                cell_fw1 = tf.contrib.rnn.DropoutWrapper(cell_fw1, output_keep_prob=keep_prob)
                cell_bw1 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                          initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113))
                cell_bw1 = tf.contrib.rnn.DropoutWrapper(cell_bw1, output_keep_prob=keep_prob)
                (enc_outputs_list, enc_state_list) = tf.nn.bidirectional_dynamic_rnn(
                          cell_fw1, cell_bw1, input_encoder, dtype=tf.float32,
                          sequence_length=input_slides_len)
                enc_outputs = tf.concat(enc_outputs_list, axis=2)

            if FLAGS.num_layers > 1: # 2 layers
                with tf.variable_scope('BLSTM2') as scope:
                    cell_fw2 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
                    cell_fw2 = tf.contrib.rnn.DropoutWrapper(cell_fw2, output_keep_prob=keep_prob)
                    cell_bw2 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113))
                    cell_bw2 = tf.contrib.rnn.DropoutWrapper(cell_bw2, output_keep_prob=keep_prob)
                    (enc_outputs_list, enc_state_list) = tf.nn.bidirectional_dynamic_rnn(
                              cell_fw2, cell_bw2, enc_outputs, dtype=tf.float32,
                              sequence_length=input_slides_len)
                enc_outputs = tf.concat(enc_outputs_list, axis=2)

            if FLAGS.num_layers > 2: # 2 layers
                with tf.variable_scope('BLSTM3') as scope:
                    cell_fw3 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
                    cell_fw3 = tf.contrib.rnn.DropoutWrapper(cell_fw3, output_keep_prob=keep_prob)
                    cell_bw3 = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                              initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113))
                    cell_bw3 = tf.contrib.rnn.DropoutWrapper(cell_bw3, output_keep_prob=keep_prob)
                    (enc_outputs_list, enc_state_list) = tf.nn.bidirectional_dynamic_rnn(
                              cell_fw3, cell_bw3, enc_outputs, dtype=tf.float32,
                              sequence_length=input_slides_len)

            encoder_state_out = tf.concat([enc_state_list[1].c,enc_state_list[1].h], axis=1) # state of the bw layer
            attention_states = tf.concat(enc_outputs_list, axis=2)

        else: # No bidirectional model
            cell = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
            cell = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.num_layers)
            attention_states, enc_state = tf.nn.dynamic_rnn(cell, input_encoder, dtype=tf.float32,
                          sequence_length=input_slides_len)
            encoder_state_out = tf.concat([enc_state[0].c, enc_state[0].h], axis=1)

        variable_summaries(encoder_state_out, 'encoder_state_out')
        variable_summaries(attention_states, 'attention_states')



    ##DECODER
    with tf.name_scope('decoder') as scope:


        W_decoder = tf.Variable(tf.truncated_normal([FLAGS.dim_lstm, num_classes], stddev=0.1), name='W_decoder')
        b_decoder = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='bias_decoder')
        variable_summaries(W_decoder, 'W_decoder')

        def loop_function(prev, _):
            # The next input are a softmax of the previous input
            relu_prev = tf.nn.relu(tf.matmul(prev, W_decoder) + b_decoder)
            return tf.nn.softmax(relu_prev)



        cell_dec = tf.contrib.rnn.LSTMCell(FLAGS.dim_lstm,
                                           initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                                           state_is_tuple=False)
        cell_dec = tf.contrib.rnn.DropoutWrapper(cell_dec, output_keep_prob=keep_prob)

        def decoder(feed_previous_bool):
            loop_f = None if feed_previous_bool else loop_function
            reuse = None if feed_previous_bool else True
            with tf.variable_scope(
                tf.get_variable_scope(), reuse=reuse) as scope:
                dec_outputs, _ = tf.contrib.legacy_seq2seq.attention_decoder(input_decoder, encoder_state_out,
                                                                      attention_states, cell_dec, num_heads=FLAGS.num_heads,
                                                                      loop_function=loop_f)
            return dec_outputs

        # If feed_previous = True --> TEST: use the previous predicted output for the next output
        # If feed_previous = False -->  TRAIN: use the real previous output to predict the next output
        feed_previous = tf.placeholder(tf.bool)
        dec_outputs = tf.cond(feed_previous, lambda: decoder(True), lambda: decoder(False))
        #print('-- dec_outputs: ', dec_outputs)



    with tf.name_scope('outputs') as scope:
        dense_outputs = [tf.nn.relu(tf.matmul(dec_o, W_decoder) + b_decoder) for dec_o in dec_outputs]
        output_proba = tf.concat([tf.expand_dims(t,1) for t in dense_outputs], 1)
        variable_summaries(dense_outputs, 'dense_outputs')
        #Prediction probs
        output = tf.concat([tf.expand_dims(tf.nn.softmax(t),1) for t in dense_outputs], 1)
        #print('-- output: ', output)


    #Loss
    with tf.name_scope('loss') as scope:
        regularized_list = ['convolutions/Variable:0','convolutions/Variable_1:0','convolutions/Variable_2:0',
                            'decoder/W_decoder:0','attention_decoder/AttnW_0:0',
                            'attention_decoder/weights:0','attention_decoder/Attention_0/weights:0',
                            'attention_decoder/AttnOutputProjection/weights:0']
        loss = tf.contrib.legacy_seq2seq.sequence_loss(dense_outputs, targets, weights, name='seq2seq')

        loss_regularized = loss + FLAGS.lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if tf_var.name in regularized_list)

        loss_summary = tf.summary.scalar("loss", loss)
        loss_regularized_summary = tf.summary.scalar("loss_regularized", loss_regularized)


    #Lists of encoder vs other vars
    encoder_vars=[]
    other_vars = []
    for t in tf.trainable_variables():
        if t.name[:7] == 'convolu':
            encoder_vars += [t]
        else:
            other_vars += [t]

    #Trainer
    with tf.name_scope('trainer') as scope:
        starter_learning_rate = FLAGS.learning_rate
        #tf.placeholder(tf.float32, name='learning_rate')
        '''
        opt_encoder = tf.train.AdamOptimizer(starter_learning_rate*FLAGS.pct_lr_char_model, beta1=0.9, beta2=0.999, epsilon=1e-08)
        opt_other   = tf.train.AdamOptimizer(starter_learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)


        grads = tf.gradients(loss_regularized, encoder_vars + other_vars)
        grads_encoder = grads[:len(encoder_vars)]
        grads_other   = grads[len(encoder_vars):]

        train_encoder = opt_encoder.apply_gradients(zip(grads_encoder, encoder_vars))
        train_other   = opt_other.apply_gradients(zip(grads_other, other_vars))
        train_op      = tf.group(train_encoder, train_other)
        '''

        global_step = tf.Variable(0, trainable=False)
        learning_rate_mdl = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       FLAGS.exponential_decay_step,
                                                       FLAGS.exponential_decay_rate,
                                                       staircase=True)
        # Passing global_step to minimize() will increment it at each step.
        train_op = tf.train.AdamOptimizer(learning_rate_mdl, beta1=0.9,
                                          beta2=0.999, epsilon=1e-08).minimize(loss_regularized,
                                                                               global_step=global_step)


    # Saver
    saver = tf.train.Saver(max_to_keep=15)

    # Summaries
    summaries_dir = data_path + 'experiments/' + FLAGS.experiment
    with tf.name_scope('summaries') as scope:
        merged = tf.summary.merge_all()

    # Add to collection
    tf.add_to_collection('input_image', input_image)
    tf.add_to_collection('input_slides_len', input_slides_len)
    tf.add_to_collection('input_word_chars', input_word_chars)
    tf.add_to_collection('input_targets', input_targets)
    tf.add_to_collection('input_weights', input_weights)
    tf.add_to_collection('output_proba', output_proba)
    tf.add_to_collection('output', output)
    tf.add_to_collection('keep_prob', keep_prob)
    tf.add_to_collection('feed_previous', feed_previous)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('train_op', train_op)
    tf.add_to_collection('merged', merged)

    for t in tf.trainable_variables():
        print(t.name)


print('MODEL CREATED!')



def decode_response(response_array, decoder_dict):
    '''
    Convert numeric codes to output sequence of text
    '''
    response_text = []
    for i in range(response_array.shape[0]):
        response_dec = [np.argmax(r) for r in response_array[i,:,:]]
        response_text += [decode_text(response_dec, decoder_dict)]
    return response_text


def train_batch(batch_size, lr=0.001, epoch=1):
    '''
    Train step
    '''
    wer_val = []
    min_wer_val = 1
    continue_training = True
    current_step = 1


    while continue_training:
        tic = time.clock()
        loss_cumm = []
        seq = batch_generator_epoch(img_trn, img_l_trn, target_trn, target_l_trn, batch_size=FLAGS.batch_size,
                                    seq_length=FLAGS.seq_decoder_len,
                                    slides_stride=FLAGS.slides_stride, x_slide_size=FLAGS.x_slide_size,
                                    num_classes=num_classes)
        for s in seq:
            feed_dict = {input_image: s[0],
                         input_slides_len: s[1],
                         input_word_chars: s[2],
                         input_targets: s[3],
                         input_weights: s[4],
                         keep_prob: FLAGS.keep_prob,
                         feed_previous: True # False feed_previous in the trainig process.
                        }
            _, loss_t = sess.run([train_op, loss], feed_dict)
            loss_cumm += [loss_t]

        # Sumaries train
        summary_str= sess.run(merged, feed_dict)
        train_writer.add_summary(summary_str, epoch)

        #Validation
        loss_cumm_tst = []
        real_tst = []
        predict_tst=[]
        predict_proba_tst=[]
        correct = 0
        num_cases = 0
        seq_tst = batch_generator_epoch(img_val, img_l_val, target_val, target_l_val,
                                    batch_size=FLAGS.batch_size, seq_length=FLAGS.seq_decoder_len,
                                    slides_stride=FLAGS.slides_stride, x_slide_size=FLAGS.x_slide_size,
                                    num_classes=num_classes, augmentation=False)
        for s_tst in seq_tst:
            feed_dict_tst = {input_image: s_tst[0],
                             input_slides_len: s_tst[1],
                             input_word_chars: s_tst[2],
                             input_targets: s_tst[3],
                             input_weights: s_tst[4],
                             keep_prob: 1,
                             feed_previous: True} # True feed_previous in the test process.
            loss_tst, out_tst, out_proba_tst, current_lr = sess.run([loss, output, output_proba, learning_rate_mdl], feed_dict_tst)
            loss_cumm_tst += [loss_tst]
            predict_proba_tst += [out_proba_tst]

            response_predict_text = decode_response(out_tst, decoder_dict)
            for resp in range(len(out_tst)):
                real_tst += [decode_text(s_tst[3][resp], decoder_dict)[:-4]]
                predict_tst += [response_predict_text[resp][:-4]]
                num_cases += 1
                if decode_text(s_tst[3][resp], decoder_dict) == response_predict_text[resp]:
                    correct += 1

        predict_proba_tst = np.concatenate(predict_proba_tst, axis=0)

        # Sumaries validation
        summary_str = sess.run(merged, feed_dict_tst)
        test_writer.add_summary(summary_str, epoch)

        print('Epoch: ',epoch, ' - LR: ',current_lr, '- Loss trn: ', np.mean(loss_cumm), ' - Loss tst: ', np.mean(loss_cumm_tst))
        print('Correct: ', correct, ' - pct correct: ', float(correct)/float(num_cases))
        print('Time', time.clock()-tic)

        # Print some results
        print('Real vs pred:')
        for resp in range(10):
            print('     ', real_tst[resp], ' vs ', predict_tst[resp])


        savefile = saver.save(sess, data_path + 'experiments/' + FLAGS.experiment +'/model-epoch', global_step=epoch)
        print('Model saved in ', savefile)


        # Early stoping criteria
        wer, cer = evaluate_corpus(real_tst, predict_tst, corpus_dict, print_results=False)
        print('VALIDATION LEXICON - WER: ',wer, ' - CER: ', cer)

        wer_val += [wer]
        if min_wer_val > wer:
            min_wer_val = wer
            # Save best model
            savefile = saver.save(sess, data_path + 'experiments/' + FLAGS.experiment +'/best_model')
            print('Model saved in ', savefile)

        if current_step>FLAGS.min_steps:
            # Stopping criteria: not improvement in last 10 epochs
            print('min_wer_val (WER):', min_wer_val)
            print('wer_val (WER) last 10:', wer_val[-10:])
            if min_wer_val < np.min(wer_val[-10:]):
                continue_training = False

        if current_step>FLAGS.max_steps:
            continue_training = False


        epoch += 1
        current_step += 1

    return real_tst, predict_tst, predict_proba_tst


# Load lexicon to early stopping
if FLAGS.train_data != 'rimes_words_48_192.hdf5': # If IAM
    corpus_dict = pickle.load(open( '/home/jorge/data/tesis/handwriting/p08_language_model/brown_lob_20000.pkl',"rb" ))
else:
    corpus_dict = pickle.load(open( '/home/jorge/data/tesis/handwriting/p08_language_model/rimes_trn_lexicon.pkl',"rb" ))


with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

    # Merge all the summaries and write them out to /tmp/mnist_logs
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test')


    # Initialize vars if dont exist previous checkpoints.
    ckpt = tf.train.get_checkpoint_state(data_path + 'experiments/' + FLAGS.experiment)
    if ckpt == None:
        # Initialize vars
        tf.global_variables_initializer().run()
        print('vars initialized!')
        epoch_ini = 1
    else:
        # Load last model
        saver.restore(sess, ckpt.model_checkpoint_path)
        if os.path.basename(ckpt.model_checkpoint_path).split('-')[-1] == 'best_model':
            epoch_ini = 1
        else:
            epoch_ini = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[-1]) + 1
        print('model loaded: ', ckpt.model_checkpoint_path)

    # Early sttoping
    real_tst, predict_tst, predict_proba_tst = train_batch(FLAGS.batch_size, lr=FLAGS.learning_rate, epoch=epoch_ini)




# Accuracy measures of the best model
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #Restore best model
    saver.restore(sess, data_path + 'experiments/' + FLAGS.experiment +'/best_model')


    #Evaluate test data
    loss_cumm_tst = []
    real_tst = []
    predict_tst=[]
    correct = 0
    num_cases = 0
    seq_tst = batch_generator_epoch(img_tst, img_l_tst, target_tst, target_l_tst,
                                    batch_size=FLAGS.batch_size, seq_length=FLAGS.seq_decoder_len,
                                    slides_stride=FLAGS.slides_stride, x_slide_size=FLAGS.x_slide_size,
                                    num_classes=num_classes, augmentation=False)
    for s_tst in seq_tst:
        feed_dict_tst = {input_image: s_tst[0],
                         input_slides_len: s_tst[1],
                         input_word_chars: s_tst[2],
                         input_targets: s_tst[3],
                         input_weights: s_tst[4],
                         keep_prob: 1,
                         feed_previous: True} # True feed_previous in the test process.
        loss_tst, out_tst = sess.run([loss, output], feed_dict_tst)
        loss_cumm_tst += [loss_tst]

        response_predict_text = decode_response(out_tst, decoder_dict)
        for resp in range(len(out_tst)):
            real_tst += [decode_text(s_tst[3][resp], decoder_dict)[:-4]]
            predict_tst += [response_predict_text[resp][:-4]]
            num_cases += 1
            if decode_text(s_tst[3][resp], decoder_dict) == response_predict_text[resp]:
                correct += 1


    # Save real and predict test words
    with open(data_path + 'experiments/' + FLAGS.experiment +'/test_words_real.txt', 'w') as fp:
        for item in real_tst:
            fp.write("%s\n" % item)
    with open(data_path + 'experiments/' + FLAGS.experiment +'/test_words_pred.txt', 'w') as fp:
        for item in predict_tst:
            fp.write("%s\n" % item)


    #Score test data with several dictionaries
    print('=============================')
    WER, CER = calculate_wer_cer(real_tst, predict_tst, predict_tst, print_results=True)
    print(' No lexicon errors:')
    print('WER: ', WER, ' - CER: ',CER)

    print('=============================')
    real_tst_dictionary = get_real_test_dictionary_set(data_path + FLAGS.train_data)
    wer, cer = evaluate_corpus(real_tst, predict_tst, real_tst_dictionary, print_results=True)
    print('Test lexicon errors:')
    print('WER: ',wer, ' - CER: ', cer)

    #Load corpus dictionary and evaluate CER and WER
    if FLAGS.train_data != 'rimes_words_48_192.hdf5': # If IAM
        print('=============================')
        corpus_dict = pickle.load(open( '/home/jorge/data/tesis/handwriting/p08_language_model/brown_lob_20000.pkl',"rb" ))
        wer, cer = evaluate_corpus(real_tst, predict_tst, corpus_dict, print_results=True)
        print('Brown - LOB lexicon errors:')
        print('WER: ',wer, ' - CER: ', cer)
    else:
        print('=============================')
        corpus_dict = pickle.load(open( '/home/jorge/data/tesis/handwriting/p08_language_model/rimes_trn_lexicon.pkl',"rb" ))
        wer, cer = evaluate_corpus(real_tst, predict_tst, corpus_dict, print_results=True)
        print('RIMES train:')
        print('WER: ',wer, ' - CER: ', cer)

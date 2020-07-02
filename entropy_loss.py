from __future__ import print_function

import os
import sys
import numpy as np
from scipy.special import softmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
# import tensorflow_probability as tfp

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from keras.datasets import mnist

# tfd = tfp.distributions

flags = tf.app.flags
flags.DEFINE_float('entropy_coeff', 10.0,
                   'Coefficient for MMCE error term.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')

flags.DEFINE_integer('num_epochs', 15, 'Number of epochs of training.')

FLAGS = flags.FLAGS

BASE_DIR = '' #/Users/xuanchen/Desktop/adversarial/MMCE
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

num_validation_samples = 1000
save_model = False
load_model = False
x_pval = x_train[-num_validation_samples:]
y_pval = y_train[-num_validation_samples:]

x_val = x_test
y_val = y_test


def get_out_tensor(tensor1, tensor2):
    return tf.reduce_mean(tensor1*tensor2)

def calibration_unbiased_loss(logits, correct_labels):
    """Function to compute MMCE_m loss."""  
    predicted_probs = tf.nn.softmax(logits)
    pred_labels = tf.argmax(predicted_probs, 1)
    predicted_probs = tf.reduce_max(predicted_probs, 1)
    correct_mask = tf.where(tf.equal(pred_labels, correct_labels),
                            tf.ones(tf.shape(pred_labels)),
                            tf.zeros(tf.shape(pred_labels)))
    c_minus_r = tf.to_float(correct_mask) - predicted_probs
    dot_product = tf.matmul(tf.expand_dims(c_minus_r, 1),
                            tf.transpose(tf.expand_dims(c_minus_r, 1)))
    tensor1 = predicted_probs
    prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                                [1, tf.shape(tensor1)[0]]), 2)
    prob_pairs = tf.concat([prob_tiled, tf.transpose(prob_tiled, [1, 0, 2])],
                            axis=2)

    def tf_kernel(matrix):
        return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))  

    kernel_prob_pairs = tf_kernel(prob_pairs)
    numerator = dot_product*kernel_prob_pairs
    return tf.reduce_sum(numerator)/tf.square(tf.to_float(tf.shape(correct_mask)[0]))
        

def self_entropy(probs):
    log_logits = tf.log(probs + 1e-10)
    logits_log_logits = probs*log_logits
    return -tf.reduce_mean(logits_log_logits,axis = 1)*10

def model(inputs):

    ''' Generate the lenet model '''

    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
    padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT") 

    conv1 = tf.layers.conv2d(
          inputs=padded_input,
          filters=6, # Number of filters.
          kernel_size=5, # Size of each filter is 5x5.
          padding="valid", # No padding is applied to the input.
          activation=tf.nn.relu,
          name='conv_layer1')

    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
          inputs=pool1,
          filters=16, # Number of filters
          kernel_size=5, # Size of each filter is 5x5
          padding="valid", # No padding
          activation=tf.nn.relu)
    # Reshaping output into a single dimention array for input to fully connected layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    # Fully connected layer #1: Has 120 neurons
    dense1 = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)

    # Fully connected layer #2: Has 84 neurons
    dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.relu)

    # Output layer, 10 neurons for each digit
    probs = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.softmax)
    
    return probs
def cal_uncertainty_error(logits, true_labels):
    predicted_probs = logits
    pred_labels = tf.argmax(predicted_probs, 1)
    incorrect_mask = tf.where(tf.equal(true_labels, pred_labels),
                            tf.zeros(tf.shape(true_labels)),
                            tf.ones(tf.shape(true_labels)))
    # entropy = tfd.Categorical(probs=predicted_probs).entropy()
    correct_mask = tf.where(tf.equal(true_labels, pred_labels),
                            tf.ones(tf.shape(true_labels)),
                            tf.zeros(tf.shape(true_labels)))
    entropy = self_entropy(predicted_probs)

    entropy_penalty = tf.reduce_sum(incorrect_mask*entropy)
    entropy_correct = tf.reduce_sum(correct_mask*entropy)
    return tf.stop_gradient(entropy_penalty)-tf.stop_gradient(entropy_correct)

def ce_loss(logits, true_labels):
    return tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.log(logits+1e-10),
                                                     labels=true_labels)) 
    
def add_loss(logits, true_labels):

    uncertainty_error = 1.0*cal_uncertainty_error(logits, true_labels)
    ce_error = tf.reduce_mean(
      tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.log(logits+1e-10),
                                                     labels=true_labels)) 
    return ce_error - FLAGS.entropy_coeff*uncertainty_error

def optimize(loss):
    opt = tf.train.AdamOptimizer()
    train_opt = opt.minimize(loss)
    return train_opt

tf.compat.v1.disable_eager_execution()
input_placeholder = tf.placeholder(tf.float32, [None, 28, 28], name="input")
input_labels = tf.placeholder(tf.int64, [None, ], name="label")


logits_layer = model(input_placeholder)
loss_layer = add_loss(logits_layer, input_labels)
train_op = optimize(loss_layer)
entropyloss_layer = cal_uncertainty_error(logits_layer, input_labels)
ce_layer = ce_loss(logits_layer, input_labels)


predictions = tf.argmax(logits_layer, 1)
acc = tf.reduce_sum(tf.where(tf.equal(predictions, input_labels),
                    tf.ones(tf.shape(predictions)),
                    tf.zeros(tf.shape(predictions))))

if not load_model:
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    saver = tf.train.Saver()

    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs

    total_loss = []
    ce_loss = []
    entropy_losses = []


    for epoch in range(num_epochs):

        num_samples = x_train.shape[0]
        num_batches = (num_samples // batch_size) + 1
        i = 0

        overall_avg_loss = 0.0
        overall_acc = 0.0
        overall_entropy_loss = 0.0
        overall_ce_loss = 0.0
        while i < num_samples:

            batch_x = x_train[i:i+batch_size,:]
            batch_y = y_train[i:i+batch_size]
            feed_dict = dict()
            feed_dict[input_placeholder] = batch_x
            feed_dict[input_labels] = batch_y

            entropy_loss = sess.run(entropyloss_layer, feed_dict=feed_dict)
            celoss = sess.run(ce_layer, feed_dict=feed_dict)
            loss, _, acc_train = sess.run([loss_layer, train_op, acc],
                                          feed_dict=feed_dict) 
            overall_avg_loss += loss
            overall_acc += acc_train
            overall_entropy_loss += entropy_loss
            overall_ce_loss += celoss


            i += batch_size
        print('epoch %d:' %(epoch+1))    
        print ('Train Acc: ', overall_acc/x_train.shape[0])
        print ('Train Loss: ', overall_avg_loss)
        print('Entropy Loss: ', overall_entropy_loss)
        print('CE Loss: ', overall_ce_loss)

        feed_dict = dict()
        feed_dict[input_placeholder] = x_pval
        feed_dict[input_labels] = y_pval
        accuracy, val_loss = sess.run([acc, loss_layer], feed_dict=feed_dict)
        print ('Val Accuracy: ', accuracy/x_pval.shape[0])
        print ('Val Loss: ', val_loss)
        print('-'*40)

        total_loss.append(overall_avg_loss)
        ce_loss.append(overall_ce_loss)
        entropy_losses.append(overall_entropy_loss)

    np.save('./loss/total_loss.npy', total_loss)
    np.save('./loss/ce_loss.npy', ce_loss)
    np.save('./loss/entropy_loss.npy', entropy_losses)
else:
    sess = tf.Session()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    saver = tf.train.Saver()
    saver.restore(sess, "./models_entropyloss/coeff-4.0")
    print("Model restored.")
    
    
    
if save_model:
    save = os.path.join('models_entropyloss')
    if os.path.exists(save):
        save_path = saver.save(sess, save+'/coeff-'+str(FLAGS.entropy_coeff))
    else:
        os.makedirs(save)
        save_path = saver.save(sess, save+'/coeff-'+str(FLAGS.entropy_coeff))
    print("Model saved in path: %s" % save_path)
# Final testing after training, also print the targets and logits
# for computing calibration.
feed_dict = dict()
feed_dict[input_placeholder] = x_test
feed_dict[input_labels] = y_test
accuracy, logits = sess.run([acc, logits_layer], feed_dict=feed_dict)
print('Test Accuracy: ',accuracy/x_test.shape[0])


# evaluate on rotated 60 mnist dataset
rot60_x = np.load('./RotNIST-master/data/train_x_60.npy')
rot60_y = np.load('./RotNIST-master/data/train_y_60.npy')

feed_dict = dict()
feed_dict[input_placeholder] = rot60_x
feed_dict[input_labels] = rot60_y
accuracy, logits = sess.run([acc, logits_layer], feed_dict=feed_dict)
print('Rotate 60 Accuracy: ', accuracy/rot60_x.shape[0])
np.save('./newentropy_probs/newentropy_rot60_'+str(FLAGS.entropy_coeff)+'_probs.npy', logits.tolist())


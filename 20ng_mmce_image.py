from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from tensorflow.contrib.layers import flatten

import tensorflow as tf
# import tensorflow.compat.v1 as tf
from keras.datasets import mnist

flags = tf.app.flags
flags.DEFINE_float('mmce_coeff', 1.0,
                   'Coefficient for MMCE error term.')
flags.DEFINE_integer('batch_size', 128, 'Batch size for training.')

flags.DEFINE_integer('num_epochs', 10, 'Number of epochs of training.')

FLAGS = flags.FLAGS

save_model = False
load_model = True

(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)

num_validation_samples = 1000

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
        
def self_entropy(logits):
    probs = tf.nn.softmax(logits)
    log_logits = tf.log(probs + 1e-10)
    logits_log_logits = probs*log_logits
    return -tf.reduce_mean(logits_log_logits)

def calibration_mmce_w_loss(logits, correct_labels):
    """Function to compute the MMCE_w loss."""
    predicted_probs = tf.nn.softmax(logits)
    range_index = tf.to_int64(tf.expand_dims(tf.range(0,
                                              tf.shape(predicted_probs)[0]), 1))
    predicted_labels = tf.argmax(predicted_probs, axis=1)
    gather_index = tf.concat([range_index,
                              tf.expand_dims(predicted_labels, 1)], axis=1)
    predicted_probs = tf.reduce_max(predicted_probs, 1)
    correct_mask = tf.where(tf.equal(correct_labels, predicted_labels),
                            tf.ones(tf.shape(correct_labels)),
                            tf.zeros(tf.shape(correct_labels)))
    sigma = 0.4

    def tf_kernel(matrix):
        """Kernel was taken to be a laplacian kernel with sigma = 0.4."""
        return tf.exp(-1.0*tf.abs(matrix[:, :, 0] - matrix[:, :, 1])/(sigma))  

    k = tf.to_int32(tf.reduce_sum(correct_mask))
    k_p = tf.to_int32(tf.reduce_sum(1.0 - correct_mask))
    cond_k = tf.where(tf.equal(k, 0), 0, 1)
    cond_k_p = tf.where(tf.equal(k_p, 0), 0, 1)
    k = tf.maximum(k, 1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
    k_p = tf.maximum(k_p, 1)*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                              (tf.shape(correct_mask)[0] - 2))
    correct_prob, _ = tf.nn.top_k(predicted_probs*correct_mask, k)
    incorrect_prob, _ = tf.nn.top_k(predicted_probs*(1 - correct_mask), k_p)

    def get_pairs(tensor1, tensor2):
        correct_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                          [1, tf.shape(tensor1)[0]]), 2)
        incorrect_prob_tiled = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1),
                          [1, tf.shape(tensor2)[0]]), 2)
        correct_prob_pairs = tf.concat([correct_prob_tiled,
                          tf.transpose(correct_prob_tiled, [1, 0, 2])],
                          axis=2)
        incorrect_prob_pairs = tf.concat([incorrect_prob_tiled,
                        tf.transpose(incorrect_prob_tiled, [1, 0, 2])],
                        axis=2)
        correct_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor1, 1),
                            [1, tf.shape(tensor2)[0]]), 2)
        incorrect_prob_tiled_1 = tf.expand_dims(tf.tile(tf.expand_dims(tensor2, 1),
                            [1, tf.shape(tensor1)[0]]), 2)
        correct_incorrect_pairs = tf.concat([correct_prob_tiled_1,
                      tf.transpose(incorrect_prob_tiled_1, [1, 0, 2])],
                      axis=2)
        return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

    correct_prob_pairs, incorrect_prob_pairs,\
                  correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)
    correct_kernel = tf_kernel(correct_prob_pairs)
    incorrect_kernel = tf_kernel(incorrect_prob_pairs)
    correct_incorrect_kernel = tf_kernel(correct_incorrect_pairs)  
    sampling_weights_correct = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1),
                              tf.transpose(tf.expand_dims(1.0 - correct_prob, 1)))
    correct_correct_vals = get_out_tensor(correct_kernel,
                                                      sampling_weights_correct)
    sampling_weights_incorrect = tf.matmul(tf.expand_dims(incorrect_prob, 1),
                              tf.transpose(tf.expand_dims(incorrect_prob, 1)))
    incorrect_incorrect_vals = get_out_tensor(incorrect_kernel,
                                                      sampling_weights_incorrect)
    sampling_correct_incorrect = tf.matmul(tf.expand_dims(1.0 - correct_prob, 1),
                              tf.transpose(tf.expand_dims(incorrect_prob, 1)))
    correct_incorrect_vals = get_out_tensor(correct_incorrect_kernel,
                                                      sampling_correct_incorrect)
    correct_denom = tf.reduce_sum(1.0 - correct_prob)
    incorrect_denom = tf.reduce_sum(incorrect_prob)
    m = tf.reduce_sum(correct_mask)
    n = tf.reduce_sum(1.0 - correct_mask)
    mmd_error = 1.0/(m*m + 1e-5) * tf.reduce_sum(correct_correct_vals) 
    mmd_error += 1.0/(n*n + 1e-5) * tf.reduce_sum(incorrect_incorrect_vals)
    mmd_error -= 2.0/(m*n + 1e-5) * tf.reduce_sum(correct_incorrect_vals)
    return tf.maximum(tf.stop_gradient(tf.to_float(cond_k*cond_k_p))*\
                                            tf.sqrt(mmd_error + 1e-10), 0.0)

def model(inputs):

    ''' Generate the lenet model '''

    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }
    input_layer = tf.reshape(inputs, [-1, 28, 28, 1])
    padded_input = tf.pad(input_layer, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT") 
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(padded_input,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    
    # Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID') 
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    
    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    # Activation.
    fc2 = tf.nn.relu(fc2)
    
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    
    return logits


def add_loss(logits, true_labels):
    mmce_error = 1.0*calibration_mmce_w_loss(logits, true_labels)
    one_hot_y = tf.one_hot(true_labels, 10)
    ce_error = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=one_hot_y))
    return ce_error + FLAGS.mmce_coeff*mmce_error

def optimize(loss):
    opt = tf.train.AdamOptimizer()
    train_opt = opt.minimize(loss)
    return train_opt

# tf.compat.v1.disable_eager_execution()
input_placeholder = tf.placeholder(tf.float32, [None, 28, 28], name="input")
input_labels = tf.placeholder(tf.int64, [None, ], name="label")


logits_layer = model(input_placeholder)
loss_layer = add_loss(logits_layer, input_labels)
train_op = optimize(loss_layer)
predicted_probs = tf.nn.softmax(logits_layer)
predictions = tf.argmax(logits_layer, 1)
acc = tf.reduce_sum(tf.where(tf.equal(predictions, input_labels),
                    tf.ones(tf.shape(predictions)),
                    tf.zeros(tf.shape(predictions))))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

saver = tf.train.Saver()

batch_size = FLAGS.batch_size
num_epochs = FLAGS.num_epochs

if not load_model:
    for epoch in range(num_epochs):

        num_samples = x_train.shape[0]
        num_batches = (num_samples // batch_size) + 1
        i = 0

        overall_avg_loss = 0.0
        overall_acc = 0.0
        while i < num_samples:

            batch_x = x_train[i:i+batch_size,:]
            batch_y = y_train[i:i+batch_size]

            feed_dict = dict()
            feed_dict[input_placeholder] = batch_x
            feed_dict[input_labels] = batch_y

            loss, _, acc_train = sess.run([loss_layer, train_op, acc],
                                          feed_dict=feed_dict)
            overall_avg_loss += loss
            overall_acc += acc_train

            i += batch_size
        print('epoch %d:' %epoch)    
        print ('Train acc: ', overall_acc/x_train.shape[0])
        print ('Train Loss: ', overall_avg_loss)

        feed_dict = dict()
        feed_dict[input_placeholder] = x_pval
        feed_dict[input_labels] = y_pval
        accuracy, val_loss = sess.run([acc, loss_layer], feed_dict=feed_dict)
        print ('Val accuracy: ', accuracy/x_pval.shape[0], 'Val Loss: ',val_loss)
        print('-'*20)
else:
    saver = tf.train.Saver()
    restorepath = "./models/model-" +str(FLAGS.mmce_coeff)
    saver.restore(sess, restorepath)
    print("Model restored from %s." %restorepath)


if save_model:
    save_path = saver.save(sess, "./models/model-"+str(FLAGS.mmce_coeff))
    print("Model saved in path: %s" % save_path)
# Final testing after training
# feed_dict = dict()
# feed_dict[input_placeholder] = x_test
# feed_dict[input_labels] = y_test
# accuracy, probs = sess.run([acc, predicted_probs], feed_dict=feed_dict)
# print('Test Accuracy: ',accuracy/x_test.shape[0])
# np.save('./mmce_probs/mmce_rot0_probs_'+str(FLAGS.mmce_coeff)+'_.npy', probs.tolist())

# evaluate on rotated 60 mnist dataset
rot60_x = np.load('./RotNIST-master/data/train_x_30.npy')
rot60_y = np.load('./RotNIST-master/data/train_y_30.npy')

feed_dict = dict()
feed_dict[input_placeholder] = rot60_x
feed_dict[input_labels] = rot60_y
accuracy, probs = sess.run([acc, predicted_probs], feed_dict=feed_dict)
print('Rotate 60 Accuracy: ', accuracy/rot60_x.shape[0])
np.save('./mmce_probs_30/mmce_rot30_probs_'+str(FLAGS.mmce_coeff)+'_.npy', probs.tolist())

# evaluate on rotated mnist dataset
# evaluate_angle = [int(i) for i in range(30, 180, 30)]
# for a in evaluate_angle:

#     rotate_x = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_x_'+str(a)+'.npy')
#     rotate_y = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_y_'+str(a)+'.npy')
    
#     feed_dict = dict()
#     feed_dict[input_placeholder] = rotate_x
#     feed_dict[input_labels] = rotate_y
#     accuracy, probs = sess.run([acc, predicted_probs], feed_dict=feed_dict)
    
#     print('Rotate %d Accuracy: %.4f' %(a, accuracy/rotate_x.shape[0]))

#     np.save('./mmce_probs/mmce_rot'+str(a)+'_probs_'+str(FLAGS.mmce_coeff)+'_.npy', probs.tolist())




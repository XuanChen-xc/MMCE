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
from real_nvp_MNIST import forward_pass, backward_pass, z_classifier, loglikelihood, getMask, int_shape
from tensorflow.contrib.layers import flatten
# import tensorflow_probability as tfp

import tensorflow as tf
# import tensorflow.compat.v1 as tf
from keras.datasets import mnist

# tfd = tfp.distributions

flags = tf.app.flags
flags.DEFINE_float('density_coeff', 0.0001,
                   'Coefficient for density error term.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for training.')

flags.DEFINE_integer('num_epochs', 8, 'Number of epochs of training.')

FLAGS = flags.FLAGS

num_validation_samples = 1000
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.astype(np.int64)
x_test = x_test.astype(np.int64)
x_val = x_train[-num_validation_samples:]
y_val = y_train[-num_validation_samples:]

save_model = True
load_model = False

def get_out_tensor(tensor1, tensor2):
    return tf.reduce_mean(tensor1*tensor2)        

def self_entropy(logits):
    probs = tf.nn.softmax(logits)
    log_logits = tf.log(probs + 1e-10)
    logits_log_logits = probs*log_logits
    return -tf.reduce_mean(logits_log_logits,axis = 1)#*10

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

def cal_density_error(logits, true_labels, log_likelihood):
    predicted_probs = tf.nn.softmax(logits)
   
    entropy = self_entropy(predicted_probs)

    return tf.stop_gradient(tf.reduce_sum(entropy)*(-1*log_likelihood))

def ce_loss(logits, true_labels):
    one_hot_y = tf.one_hot(true_labels, 10)
    return tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=one_hot_y)) 
    
def add_loss(logits, true_labels, log_likelihood):

    density_error = 1.0*cal_density_error(logits, true_labels, log_likelihood)
    one_hot_y = tf.one_hot(true_labels, 10)
    ce_error = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                     labels=one_hot_y)) 
    return ce_error + FLAGS.density_coeff*density_error

def optimize(loss):
    opt = tf.train.AdamOptimizer()
    train_opt = opt.minimize(loss)
    return train_opt

# tf.compat.v1.disable_eager_execution() # when using tf2.0

# nvp generator parameters
layer_num = 4
sample_size = 1
learning_rate = 0.0001

# create the model
generator = tf.make_template('generator', forward_pass)
inv_generator = tf.make_template('generator', backward_pass, unique_name_='generator')
classifier = tf.make_template('classifier', z_classifier)

# get loglikelihood gradients over GPU
all_params = tf.trainable_variables()
with tf.device('/gpu:0'):
    optimizer = tf.train.AdamOptimizer(
            learning_rate=0.0001,
            beta1=1. - 1e-1,
            beta2=1. - 1e-3,
            epsilon=1e-08)
    # Generator 
    x_input = tf.placeholder(tf.float32, shape=[None, 784])
    xs = tf.placeholder(tf.int32, shape=[2])
    is_training = tf.placeholder(tf.bool)
    mask = tf.placeholder(tf.float32, shape=[None, 784])
    reuse = tf.placeholder(tf.bool)
    z, jacs = generator(x_input, xs, layer_num, mask, is_training)
    log_likelihood = loglikelihood(z, jacs)
    gen_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    train_step_gen = optimizer.minimize(log_likelihood, var_list=gen_train_vars)
    print('building generator done')
    # Z classifier
    z_input = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.float32, shape=[None, 10])
    output_Z_classifier = classifier(z_input, is_training)
    cross_entropy_Z = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output_Z_classifier))
    class_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
    train_step_Z_classifier = optimizer.minimize(cross_entropy_Z, var_list=class_train_vars) 
    correct_prediction_Z = tf.equal(tf.argmax(output_Z_classifier,1), tf.argmax(labels,1))
    accuracy_Z = tf.reduce_mean(tf.cast(correct_prediction_Z, tf.float32))

# init & save

saver = tf.train.Saver()
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True


input_placeholder = tf.placeholder(tf.float32, [None, 28, 28], name="input")
input_labels = tf.placeholder(tf.int64, [None, ], name="label")


logits_layer = model(input_placeholder)
predicted_probs = tf.nn.softmax(logits_layer)
loss_layer = add_loss(logits_layer, input_labels, log_likelihood)
train_op = optimize(loss_layer)
densityloss_layer = cal_density_error(logits_layer, input_labels, log_likelihood)
ce_layer = ce_loss(logits_layer, input_labels)

predictions = tf.argmax(logits_layer, 1)
acc = tf.reduce_sum(tf.where(tf.equal(predictions, input_labels),
                    tf.ones(tf.shape(predictions)),
                    tf.zeros(tf.shape(predictions))))

sess = tf.InteractiveSession(config=config)
initializer = tf.global_variables_initializer()
sess.run(initializer)
sess.run(tf.local_variables_initializer())
# restore generator
if not load_model:
    ckpt_file = "./checkpoints/mnist_gen.ckpt"
    print('restoring generator from ', ckpt_file)
    saver.restore(sess, ckpt_file) 

if not load_model:
   
    batch_size = FLAGS.batch_size
    num_epochs = FLAGS.num_epochs

    total_loss = []
    ce_loss = []
    density_losses = []

    for epoch in range(num_epochs):

        num_samples = x_train.shape[0]
        num_batches = (num_samples // batch_size) + 1
        i = 0
        overall_avg_loss = 0.0
        overall_acc = 0.0
        overall_density_loss = 0.0
        overall_ce_loss = 0.0
        while i < num_samples:

            batch_x = x_train[i:i+batch_size,:]
            batch_y = y_train[i:i+batch_size]
            batch_x_reshape = batch_x.reshape([-1, 784])/255.
            batch_mask = getMask((batch_x_reshape.shape[0], 784)) 
            feed_dict={input_placeholder: batch_x, input_labels: batch_y, x_input: batch_x_reshape, xs: int_shape(batch_x_reshape), mask: batch_mask, is_training:True}
            density_loss = sess.run(densityloss_layer, feed_dict=feed_dict)
            celoss = sess.run(ce_layer, feed_dict=feed_dict)
            loss, _, acc_train = sess.run([loss_layer, train_op, acc],
                                          feed_dict=feed_dict) 
#             ll = sess.run(log_likelihood, feed_dict = {input_placeholder: batch_x, input_labels: batch_y, x_input: batch_x_reshape, xs: int_shape(batch_x_reshape), mask: batch_mask, is_training:False})
            overall_avg_loss += loss
            overall_acc += acc_train
            overall_density_loss += density_loss
            overall_ce_loss += celoss
           

            i += batch_size
        print('epoch %d:' %(epoch+1))    
        print ('Train Acc: ', overall_acc/x_train.shape[0])
        print ('Train Loss: ', overall_avg_loss)
        print('Density Loss: ', overall_density_loss)
        print('CE Loss: ', overall_ce_loss)
        
        x_val_reshape = x_val.reshape([-1, 784])/255.
        val_mask = getMask((x_val_reshape.shape[0], 784))
        feed_dict = {input_placeholder: x_val, input_labels: y_val, x_input: x_val_reshape, xs: int_shape(x_val_reshape), mask: val_mask, is_training: False}
        accuracy, val_loss = sess.run([acc, loss_layer], feed_dict=feed_dict)
        print ('Val Accuracy: ', accuracy/x_val.shape[0])
        print ('Val Loss: ', val_loss)
        print('-'*40)

        total_loss.append(overall_avg_loss)
        ce_loss.append(overall_ce_loss)
        density_losses.append(overall_density_loss)

    # np.save('./loss/total_loss.npy', total_loss)
    # np.save('./loss/ce_loss.npy', ce_loss)
    # np.save('./loss/density_loss.npy', density_losses)
else:
    
    restorepath = "./models_densityloss/coeff-"+str(FLAGS.density_coeff)
    saver.restore(sess, restorepath)
    print("Model restored from %s" %restorepath)
    
    
if save_model:
    save = os.path.join('models_densityloss')
    if os.path.exists(save):
        save_path = saver.save(sess, save+'/coeff-'+str(FLAGS.density_coeff))
    else:
        os.makedirs(save)
        save_path = saver.save(sess, save+'/coeff-'+str(FLAGS.density_coeff))
    print("Model saved in path: %s" % save_path)

# evaluate on different rotate mnist dataset
evaluate_angle = [int(i) for i in range(15, 195, 30)]
prediction_error = []
accuracies = []
batch_size = FLAGS.batch_size
for a in evaluate_angle:

    rotate_x = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_x_'+str(a)+'.npy')
    rotate_y = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_y_'+str(a)+'.npy')
    batch_x_reshape = rotate_x.reshape([-1, 784])/255.
    batch_mask = getMask((batch_x_reshape.shape[0], 784)) 
    feed_dict={input_placeholder: rotate_x, input_labels: rotate_y, x_input: batch_x_reshape, xs: int_shape(batch_x_reshape), mask: batch_mask, is_training:False}
    accuracy, probs = sess.run([acc, predicted_probs], feed_dict=feed_dict)
    print('Rotate %d Accuracy: %.4f' %(a, accuracy/rotate_x.shape[0]))
    np.save('./density_probs/density_rot'+str(a)+'_probs_'+str(FLAGS.density_coeff)+'_.npy', probs.tolist())
    
    
    
#     num_samples = rotate_x.shape[0]
#     num_batches = (num_samples // batch_size) + 1
#     i = 0
#     overall_acc = 0.0
#     overall_loss = 0.0  
#     overall_density_loss = 0.0
#     overall_ce_loss = 0.0
#     while i < num_samples:
   
#         batch_x = rotate_x[i:i+batch_size,:]
#         batch_y = rotate_y[i:i+batch_size]
#         batch_x_reshape = batch_x.reshape([-1, 784])/255.
#         batch_mask = getMask((batch_x_reshape.shape[0], 784)) 

#         feed_dict={input_placeholder: batch_x, input_labels: batch_y, x_input: batch_x_reshape, xs: int_shape(batch_x_reshape), mask: batch_mask, is_training:False}
#         density_loss = sess.run(densityloss_layer, feed_dict=feed_dict)
#         celoss = sess.run(ce_layer, feed_dict=feed_dict)
#         loss, acc_rotate = sess.run([loss_layer, acc],feed_dict=feed_dict) 
#         overall_loss += loss
#         overall_acc += acc_rotate
#         overall_density_loss += density_loss
#         overall_ce_loss += celoss

#         i += batch_size
    
#     print('Rotate %d Accuracy: %.4f' %(a, overall_acc/rotate_x.shape[0]))
#     prediction_error.append(overall_loss)
#     accuracies.append(overall_acc/rotate_x.shape[0])

#     np.save('./mmce_probs/mmce_rot'+str(a)+'_probs_'+str(FLAGS.mmce_coeff)+'_.npy', probs.tolist())

# np.save('./plot_error_acc/density_error_'+str(FLAGS.density_coeff)+'_.npy', prediction_error)
# np.save('./plot_error_acc/density_acc_'+str(FLAGS.density_coeff)+'_.npy', accuracies)


# evaluate on rotated 60 mnist dataset
# rot60_x = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_x_60.npy')
# rot60_y = np.load('/home/xuanc/pacman/Pacman/MMCE/RotNIST-master/data/train_y_60.npy')
# rot60_x_reshape = rot60_x.reshape([-1,784])/255.
# rot_mask = getMask((rot60_x_reshape.shape[0], 784))
# feed_dict={input_placeholder: rot60_x, input_labels: rot60_y, x_input: rot60_x_reshape, xs: int_shape(rot60_x_reshape), mask: rot_mask, is_training:False}
# accuracy, logits = sess.run([acc, logits_layer], feed_dict=feed_dict)
# print('Rotate 60 Accuracy: ', accuracy/rot60_x.shape[0])
# np.save('./density_probs/density_rot60_'+str(FLAGS.density_coeff)+'_probs.npy', logits.tolist())


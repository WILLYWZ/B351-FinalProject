# -*- coding: utf-8 -*-
"""
train CNN model
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Data_preprocessing import data_pr
import os
 
# load data and data processed
O_data,label_all = data_pr.image_segmentation()
sample_num= O_data.shape[0]
data1 = O_data/255
data_all = np.zeros(data1.shape)
for i in range(sample_num):
    data_all[i,:,:] = data1[i,:,:] - np.mean(data1[i,:,:])

data_ = np.copy(data_all)
label_ = np.copy(label_all)

#train data
rate = 0.8
T_L= int(rate * sample_num)
Train = np.zeros([T_L,50,24])
Test = np.zeros([sample_num-T_L,50,24])

Train_label = np.zeros([T_L,36])
Test_label = np.zeros([sample_num-T_L,36])
tt = sample_num-T_L
for i in range(5):
    train_index = np.arange(int(T_L/5)) + int(i*data_.shape[0]/5)
    test_index = np.arange(int((sample_num-T_L)/5))+int(T_L/5) + int(i*data_.shape[0]/5)
    Train[int(i*T_L/5):int((i+1)*T_L/5),:,:] = data_[train_index,:,:]
    Train_label[int(i*T_L/5):int((i+1)*T_L/5),:] = label_[train_index,:]
    
    Test[int(i*tt/5):int((i+1)*tt/5),:,:] = data_[test_index,:,:]
    Test_label[int(i*tt/5):int((i+1)*tt/5),:] = label_[test_index,:]
    
data = np.copy(Train)
label = np.copy(Train_label)


#------- define convolutional layer and maxpool layer--------------#
def conv_layer(x,w,b):
    conv = tf.nn.conv2d(x, w , strides = [1,1,1,1], padding = 'SAME')
    conv_with_b = tf.nn.bias_add(conv,b)
    conv_out = tf.nn.relu(conv_with_b)
    return conv_out
def maxpool_layer(conv,k=2):
    return tf.nn.max_pool(conv,ksize = [1,k,k,1],strides = [1,k,k,1],padding = 'SAME')


#--------build CNN model-------------#
# define input, output varibles
#x = tf.placeholder(tf.float32,[None,50,200])
x = tf.placeholder(tf.float32,[None,50,24])
y = tf.placeholder(tf.float32,[None,36])

def model(keep_prob = 0.5):

    # define first Convolution layer
    W1 = tf.Variable(tf.random_normal([3,3,1,64]))
    b1 = tf.Variable(tf.random_normal([64]))
    
    # define second Convolution layer
    W2 = tf.Variable(tf.random_normal([3,3,64,64]))
    b2 = tf.Variable(tf.random_normal([64]))
    
    # connected layer
    #W3 = tf.Variable(tf.random_normal([650*64,1024]))
    W3 = tf.Variable(tf.random_normal([90*64,1024]))
    b3 = tf.Variable(tf.random_normal([1024]))
    
    # output
    W_out = tf.Variable(tf.random_normal([1024,36]))
    b_out = tf.Variable(tf.random_normal([36]))
    #x_reshape = tf.reshape(x,shape = [-1,50,200,1])
    
    #----------------build model--------------#
    
    x_reshape = tf.reshape(x,shape = [-1,10,40,1])
    conv_out1 = conv_layer(x_reshape,W1,b1)
    maxpool_out1 = maxpool_layer(conv_out1)
    
    norm1 = tf.nn.lrn(maxpool_out1,4,bias = 1.0,alpha = 0.001/9.0, beta = 0.75)
    
    conv_out2 = conv_layer(norm1,W2,b2)
    norm2 = tf.nn.lrn(conv_out2,4,bias = 1.0,alpha = 0.001/9.0, beta = 0.75)
    maxpool_out2 = maxpool_layer(norm2)
    
    
    maxpool_reshape = tf.reshape(maxpool_out2,[-1,W3.get_shape().as_list()[0]]) 
    local = tf.add(tf.matmul(maxpool_reshape,W3),b3)
    local_out = tf.nn.relu(local)
    local_drop = tf.nn.dropout(local_out,keep_prob)
    out = tf.add(tf.matmul(local_drop,W_out),b_out)
    
    out = tf.add(tf.matmul(local_out,W_out),b_out)
    
    return out

def train(keep_prob = 0.5):
    model_op = model(keep_prob)
    
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits = model_op,labels = y)
    )
    
    train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
    
    correct_pred = tf.equal(tf.arg_max(model_op,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
    
    
    # Training 
    acc = []
    loss_ = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
       
        batch_size = label.shape[0] // 20
        print('batch_size = ',batch_size)
        for j in range(100):
            print('EPOCH = ', j)
            iter_ = 0
            for i in range(0,label.shape[0], batch_size):
                batch_data = data[i:i+batch_size,:]

                batch_label = label[i:i+batch_size,:]
                _,accuracy_val = sess.run([train_op,accuracy], feed_dict = {x:batch_data,y:batch_label})
               
                iter_ += 1
                
            _,loss = sess.run([train_op,cost],feed_dict = {x:batch_data,y:batch_label})
            print('Accuracy = %s, loss = %s'%(accuracy_val, loss))
            acc.append(accuracy_val)
            loss_.append(loss)
            if accuracy_val >= 0.99:
                break
        print('Done WITH EPOCH')
        t = tf.train.Saver()
        p = os.getcwd()
        base_dir = p+"\\model_trained"+"\\model.ckpt"
        t.save(sess,base_dir)
    return acc,loss_

#----------train-------------#
acc,loss_ =  train()

#plot train result

L = list(range(len(acc))) 
plt.plot(L,acc,'r')
plt.title('Epoch - Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.savefig('Epoch.png')
plt.show()

plt.plot(L,loss_,'g')
plt.title('Epoch - Loss')
plt.xlabel('Loss')
plt.ylabel('Loss')
plt.savefig('Loss.png')
plt.show()
    
    

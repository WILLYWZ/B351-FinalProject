# -*- coding: utf-8 -*-
"""
An example of CNN identifying a 
verification code picture
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from Data_preprocessing import data_pr
from PIL import Image
 
# laod data and data processed
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

        print('Done WITH EPOCH')
        t = tf.train.Saver()
        t.save(sess,'C:/model_trained/model.ckpt')
    
def test_(test,test_label):
    model_op = model(keep_prob = 1)
    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        saver.restore(sess,'C:/model_trained/model.ckpt')
        correct_pred = tf.equal(tf.arg_max(model_op,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
        test_accuracy = sess.run(accuracy,feed_dict = {x:test,y:test_label})
        #out = sess.run(model_op,feed_dict = {x:test})
    return test_accuracy

def example(test_picture):
    data = np.copy(test_picture)
    T = np.copy(data[:,30:150])
    data_cut = np.zeros([5,50,24])
    for i in range(5):
        data_cut[i,:,:] = T[:,i*24:(i+1)*24]
    data_cut1 = data_cut/255
    data_e = np.zeros(data_cut1.shape)
    for i in range(5):
        data_e[i,:,:] = data_cut1[i,:,:] - np.mean(data_cut1[i,:,:])
    model_op =  model(keep_prob = 1)
    saver = tf.train.Saver()
    with tf.compat.v1.Session() as sess:
        p = os.getcwd()
        base_dir = p+"\\model_trained"+"\\model.ckpt"
        #saver.restore(sess,'D:/程序文件/python_spyder/1911-C-1403/model_trained/model.ckpt')
        saver.restore(sess,base_dir)
        Index = tf.arg_max(model_op,1)
        Ind = sess.run(Index, feed_dict = {x:data_e})
    return Ind

def load_test_picture(i:int):
    p = os.getcwd()
    base_dir = p+"\\samples"
    filename = os.listdir(base_dir)
    im = Image.open(base_dir+ "/" + filename[i])
    L = im.convert('L')
    z = np.array(L)
    lab = filename[i]
    label = data_pr.tex_to_vector(lab)
    return z,label
def ind_to_tex(Ind):
    char_set = ['0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','j',
                    'k','l','m','n','o','p','q','r','s','t',
                    'u','v','w','x','y','z']
    text = ''
    for i in range(5):
        text = text + char_set[Ind[i]]
    return text
if __name__ == '__main__':
    # input 100th Captcha picture
    i = 1
    z,_ = load_test_picture(i)
    Ind = example(z)
    text = ind_to_tex(Ind)
    print('Captcha picture')
    plt.imshow(z)
    plt.show()
    print("Recognition :",text)
    
    
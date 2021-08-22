# -*- coding: utf-8 -*-
"""
Data import and preprocessing

"""
import numpy as np
import os
from PIL import Image

class data_pr:
    def load_data():
        p = os.getcwd()
        base_dir = p+"\\samples"
        filename = os.listdir(base_dir)
        data = []
        label = []
        for i in filename:
           
            im = Image.open(base_dir+ "/" + i)
            L = im.convert('L')
            #T=L.resize((40, 10))
            data.append(np.array(L))
            #data.append(np.array(T))
            label.append(i)
        return np.array(data),label
    def tex_to_vector(text):
        Tex = text[0:5]
        vector = np.zeros((5,36))
        for i in range(5):
            temp = ord(Tex[i])
            if temp <= 57 and temp >= 48:
                index = temp - 48
            if temp <= 122 and temp>=97:
                index = temp - 97 + 10
            vector[i,index] = 1
        return vector
    def vector_to_tex(vector):
        char_set = ['0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','j',
                    'k','l','m','n','o','p','q','r','s','t',
                    'u','v','w','x','y','z']
        index = np.where(vector == np.max(vector))
        return char_set[index[0][0]]
    def image_segmentation():
        data,label = data_pr.load_data()
        T = np.copy(data[:,:,30:150])
        L = data.shape[0]
        data_cut = np.zeros([L*5,50,24])
        for i in range(5):
            data_cut[i*L:(i+1)*L,:,:] = T[:,:,i*24:(i+1)*24]
        
        label_cut = np.zeros([L*5,36])
        for i in range(L):
            temp_lab = data_pr.tex_to_vector(label[i])
            for k in range(5):
                label_cut[k*L+i,:] = temp_lab[k,:]
        return data_cut,label_cut
    
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
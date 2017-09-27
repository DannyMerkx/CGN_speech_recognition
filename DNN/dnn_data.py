#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:05:37 2017

@author: danny
"""
import numpy as np
import math
import pickle
import os
# contains the function to handle the h5 data file for the DNNs.
# to do: implement a way to load data into memory rather than working with indices for use on smaller datasets
def Split_dataset(f_nodes, l_nodes, splice_size, index_loc, split_loc, size):
# split the data into train test and val sets. For data too big to fit in memory 
# we create only an index for the minibatcher load the data per minibatch
    train_index = []
    test_index = []
    val_index = []
    offset = 0
    # If an index was already created in a previous run just load the old ones
    try:
        
        f = open(os.path.join(index_loc, 'train_index.py'),'rb')
        print ('loading indices')
        x = True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    train_index.append(y)
            except:
                x = False

        f = open(os.path.join(index_loc, 'test_index.py'),'rb')
        x = True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    test_index.append(y)
            except:
                x = False

        f = open(os.path.join(index_loc, 'val_index.py'),'rb')
        x = True
        while x:
            try:
                temp = pickle.load(f)
                for y in temp:
                    val_index.append(y)
            except:
                x = False
    except:
    # index triple, first number is the index of each frame. Because different wav files are stored in different
    #leaf nodes of the h5 file, we also keep track of the node number and the index of the frame internal to the node.
        # open files for the train test and val index
        train = open(os.path.join(index_loc, 'train_index.py'), 'wb')
        test = open(os.path.join(index_loc, 'test_index.py'), 'wb')
        val = open(os.path.join(index_loc, 'val_index.py'), 'wb')
        # load the split file, this file should say for each wav file whether it belongs to train test or val
        split_info = np.load(split_loc)
        print('creating index')
        for x in range (0,len(f_nodes)):
            temp = []
            # get the split this node belongs to from the split.npy file
            split = [i[1] for i in split_info if i[0] == f_nodes[x].name][0]
            for y in range (splice_size, len(f_nodes[x]) - splice_size):
            # 999 was used to indicate out of vocab values. These are removed
            # from the training data, however they are still valid for splicing
            # with a valid training frame
                if l_nodes[x][y][1]!=b'999':
                    temp.append((y + offset, x, y))
            # shuffle the frames of each wav file        
            np.random.shuffle(temp)
            offset=offset+len(f_nodes[x])
            # dump this files indices in the appropriate split
            if split == 'train':
                pickle.dump(temp, train)
                for i in temp:
                    train_index.append(i)
            elif split == 'test':
                pickle.dump(temp, test)
                for i in temp:
                    test_index.append(i)
            elif split == 'val':
                pickle.dump(temp, val)
                for i in temp:
                    val_index.append(i)
        train.close()
        test.close()
        val.close()
    # get length of each part of the dataset
    train_size = len(train_index)
    test_size = len(test_index)
    val_size = len(val_index)
    print(str(train_size) + ' ' + str(test_size) + ' ' + str(val_size))
    
    #split in train, validation and test and optionally take only a percentage of the total data.    
    train_size = int(math.floor(train_size*size))
    val_size= int(math.floor(val_size*size))
    test_size = int(math.floor(test_size*size))
    Train_index = train_index[0:train_size]
    Val_index = val_index[0:val_size]  
    Test_index = test_index[0:test_size]

    return (Train_index, Val_index, Test_index)

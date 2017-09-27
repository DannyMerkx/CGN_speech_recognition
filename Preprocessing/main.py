#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:19:53 2017

@author: danny
"""
# this script functions as a config file where you set the variables for the 
# feature creation and pass it to the function that actually does the work.

from process_data import features, features_and_labels
# regex for checking the file extensions
f_ex = 'fn[0-9]+.' 
# option for the features to return, can be raw for the raw frames, 
# freq_spectrum for the fft transformed frames, fbanks for filterbanks or mfcc for mfccs. 
feat = 'fbanks'
# set data path for label files
l_path = "/data/Finetracker/transcripts"
# set data path for audio files
a_path = "/data/comp-o/nl"
#files to save the features and labels in
data_loc = "/data/processed/" + feat +".h5" #'/data/processed/mfcc.h5'
# some parameters for mfcc creation
params = []
# set alpha for the preemphasis
alpha = 0.97
# set the number of desired filterbanks
nfilters = 64
# windowsize and shift in seconds
t_window = .025
t_shift = .005

# option to include delta and double delta features
use_deltas = False
# option to include frame energy
use_energy = False
# put paramaters in a list
params.append(alpha)
params.append(nfilters) 
params.append(t_window)
params.append(t_shift)
params.append(feat)
params.append(data_loc)
params.append(use_deltas)
params.append(use_energy)
# call the function that actually does the work
features_and_labels(f_ex, params, l_path, a_path)
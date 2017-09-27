#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:21:45 2016

@author: danny
"""
import numpy as np
import math
import codecs
# functions for cleaning the transcript and labelling the data and converting
# phoneme start and end times from seconds to frames  
def parse_transcript (loc, fs):
    with codecs.open(loc, 'rb', encoding= 'latin-1') as file:
        raw_trscrpt = file.read()
        split_trscrpt = raw_trscrpt.splitlines()
    #split the lines with the articulatory features
    final_trscrpt=[]
    for x in range(0,len(split_trscrpt),3):
        final_trscrpt.append([float(split_trscrpt[x])*fs, float(split_trscrpt[x+1])*fs, split_trscrpt[x+2].split()])
    return (final_trscrpt)
    

def label_frames (nframes, trans_labels, frameshift):
    # labels the data frames, input is #frames, 
    #labelled data list
    data_labels = []
    # calculate begin and end samples of the frames
    t = [[t*frameshift, (t*frameshift)+frameshift] for t in range (0, nframes)]
    for t in t:
        # if  the begin sample# and end sample nr of the frame are smaller
        # than the end sample nr of the phoneme, the frame is fully overlapping
        # only one phoneme, and gets that label.
        if t[0] < trans_labels[0][1] and t[1] < trans_labels[0][1]:
            data_labels.append(trans_labels[0][2])
        # however if the begin sample nr is smaller but the end sample nr is larger
        # than the end sample nr of the phoneme, the frame is partially on 2 phonemes
        elif t[0] < trans_labels[0][1] and t[1] >= trans_labels[0][1]:
            # so the frame gets the label of the phoneme which more than half of the
            # frame is overlapping (can easily be checked by looking at the middle 2
            # samples). In case of a tie, the frame gets the label of the first phoneme.
            if math.floor((t[0]+t[1]-1)/2) < trans_labels[0][1]:
                data_labels.append(trans_labels[0][2])
                trans_labels.pop(0)
            elif math.floor((t[0]+t[1]-1)/2) >= trans_labels[0][1]:
                trans_labels.pop(0)
                data_labels.append(trans_labels[0][2])
    return(data_labels)
    

            

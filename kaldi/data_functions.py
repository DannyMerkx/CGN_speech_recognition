#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:07:35 2017

@author: danny
"""
import re
import os
# functions for loading the datafiles
def list_files (datapath):
    # lists all file names in the give directory
    input_files= [x for x in os.walk(datapath)]
    input_files= input_files[0][2]              
    return (input_files)
            
def phoneme_dict(loc):
    # creates a phoneme/articulatory feature (af) dictionary for mapping
    # cgn phonemes to a costum phoneme set and af set. tailored to the
    # layout of a specific conversion table file, this will NOT work on anything 
    # else without alterations

    conv_table= [x.split() for x in open (loc)]
    conv_table = [x for x in conv_table if x]            
    cgn = [x[8:] for x in conv_table if x] 

    cgndict={}
    for x in range (1,len(cgn)):
        if len(cgn[x]) ==1:
            cgndict[cgn[x][0]] = conv_table[x][0:8]
        else:
            for y in range (0,len(cgn[x])):
                cgndict[cgn[x][y]] = conv_table[x][0:8]
    return (cgndict)
    
def check_files (audio_files, label_files, f_ex):
    # checks if the file lists for the audio and transcripts match
    # properly
    regex= re.compile(f_ex)
    match=[regex.search(x) for x in audio_files]
    af=[]
    for x in range (0, len(audio_files)):
        af.append(audio_files[x][match[x].span()[0]:match[x].span()[1]])       
    match=[regex.search(x) for x in label_files]
    lf=[]
    for x in range (0, len(label_files)):
        lf.append(label_files[x][match[x].span()[0]:match[x].span()[1]])
    return(af==lf)
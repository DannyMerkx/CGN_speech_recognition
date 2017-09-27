#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:07:35 2017

@author: danny
"""
import re
import os
# functions for getting the lists of audio and transcript files, and creating 
# the phone to articulatory features conversion dictionary
def list_files (datapath):
    # lists all file names in the give directory
    input_files= [x for x in os.walk(datapath)]
    input_files= input_files[0][2]              
    return (input_files)
               
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
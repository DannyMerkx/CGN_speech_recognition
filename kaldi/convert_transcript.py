#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:12:47 2017

@author: danny
"""
# this script reads the alignment files from kaldi and extracts 
# phonemes and their start and end times, to create a transcript 
# simmilar to the CGN awd files.
import codecs
from data_functions import list_files
from label_func import parse_transcript
# location of your split alignment files created by splitAlignments.py
file_loc = "/scratch/danny/kaldi/egs/myexp/exp/tri4a_ali/split_ali/"

# we need to retrieve the end times of the CGN files, as we gave kaldi only non 
# silence utterances. Most files end in silence so we need to fill the ends of the files.
cgntrans_loc ="/scratch/danny/CGN/data/annot/text/awd/comp-o/nl/"
input_files = list_files(cgntrans_loc)
input_files.sort()

end_time=[]
for x in input_files:
    pattern= '"N[0-9]+_SEG"'
    cgntrans = parse_transcript(pattern,cgntrans_loc+x)
    end_time.append(round(float(cgntrans[-2]),4))
# list all input files
input_files = list_files(file_loc)
input_files.sort()
# set a counter to retrieve proper end times for the files
count=0
# read the data we need from the alignment files.
for f in input_files:
    file= codecs.open(file_loc+f)
    trans =[]
    for line in file:
        line=line.split()
        # column 6 holds phonemes by name, 2 contains their kaldi ids if needed
        phone = line[6]
        # calculate phoneme start time. We want the start time relative to the whole file
        # so we take the utterance start time (col 7) plus start time of the phoneme WITHIN the utterance (col 4)
        start = round(float(line[7])+float(line[4]),4)
        # end time is start time plus phoneme duration
        end = round(start + float(line[5]),4)    
        trans.append([start,end,phone])
    file.close()
    trans.sort()
    
    
    
    # it might be that the kaldi alignment does not cover the complete duration of the file
    # (because kaldi did not model silent utterances for us), in which case we need to add 
    # silence to fill the transcript. Most files start and end with sil which kaldi did not model.
    
    # add silence if the end of the transcript does not match file end time
    if trans[-1][1] != end_time[count]:
        trans.append([trans[-1][1],end_time[count],'sil'])
    count=count+1
    # add silence if the transcript does not start at 0
    if trans[0][0]!= 0:
        trans.append([0,trans[0][0],'sil'])
    trans.sort()
    # since we did not include utterances which are completely silent,
    # we fill in the gaps in our transcript with silence
    for x in range(1,len(trans)):
        try:
            if trans[x][0] != trans[x-1][1]:
                trans.append([trans[x-1][1],trans[x][0],'sil'])          
        except:
            pass
    # now we remove some annotation from the phoneme names (_B _I and _E for begin in and end)
    # furthermore, it might happen that an utterance ending in silence is followed by an utterance 
    # beginning with silence. We can concatenate these cases into one silence phoneme
    index=[]
    trans.sort()
    for x in range(0,len(trans)):
        if '_' in trans[x][2]:
            trans[x][2]=trans[x][2][:-2]
        # if two sil in a row, replace start time of the last on with 
        # that of the first one and keep track of the index.
        if x>0 and trans[x][2]=='sil' and trans[x-1][2]=='sil':
            trans[x][0] = trans[x-1][0]
            index.append(x-1)
        # now remove unneeded sil phonemes
    trans = [trans[x] for x in range (0,len(trans)) if x not in index ] 
    file= codecs.open(file_loc+f,'w')
    for t in trans:
        for x in t:
            try:
                x= str(round(x,4))
            except:
                pass
            file.write(x+'\n')

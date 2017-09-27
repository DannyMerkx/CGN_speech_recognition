#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 19:49:41 2017

@author: danny
"""
import re 
import os
import codecs
import gzip
import numpy

# this file is for preprocessing the transcripts. 

def cleanup (text, pattern):
    #cleans up the raw text file
    match = pattern.search(text)
    begin = match.span()[0]
    cl_text = text[begin:-1]
    return cl_text

def phone2af(phone_trans, cgndict):
    # convert the cleaned phonetic transcript to articulatory features
    af_trans = list(phone_trans)
    # this is a check to see if the transcript was properly cleaned and formatted
   
    # set the default phone as <oov> just in case an unknown phoneme is found.
    default = cgndict['<oov>']
    for x in range (2, len(phone_trans), 3):
        # convert the phone label to articulatory features
        artic_feat = (cgndict.get(phone_trans[x], default))
        label = '' 
        for af in artic_feat:
            label = label + af + ' '
        af_trans[x] = label
    return af_trans

def phoneme_dict(table_loc):
    # creates a phoneme/articulatory feature (af) dictionary for mapping
    # cgn phonemes to a costum phoneme set and af set. tailored to the
    # layout of a specific conversion table file, this will NOT work on anything 
    # else without alterations

    conv_table= [x.split() for x in open (table_loc)]
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

def prep_cgn(conv_table, pattern, in_path, out_path):
# prepare original cgn transcripts for labelling of training data.
# removes headers etc. and converts phonemes to articulatory features.
    paths = os.listdir(in_path)        
    regex = re.compile(pattern)
    for file in paths:
        f = os.path.join(in_path, file)
        try:
            with gzip.open(f, 'rb') as trans:
                raw_trscrpt = trans.read().decode('latin-1') 
        except:
            with codecs.open(f, 'rb', encoding= 'latin-1') as trans:
                raw_trscrpt = trans.read()

        # awd transcripts have an orthographic and phonetic part, remove the part we do not need.
        cleaned_trscrpt = cleanup(raw_trscrpt, regex)    
        # split the transcript in seperate lines
        split_trscrpt = cleaned_trscrpt.splitlines()
        # replace CGN silence "" notation with 'sil'
        for x in range(0, len(split_trscrpt)):
            split_trscrpt[x] = split_trscrpt[x].replace('""', 'sil').replace('"', '')
        # get the size of transcript from the header
        trscpt_size = int(split_trscrpt[3])*3
        # at last remove some empty lines and header lines so only the phonemes and their
        # start and end time are left
        final_trscrpt = split_trscrpt[4: trscpt_size+4]
        # transcripts should be formatted like; begin time, end time, phone. a simple sanity
        # check is to see if your transcript is divisible by 3
        af_trscrpt = phone2af(final_trscrpt, cgn_dict)
        if not numpy.mod(len(af_trscrpt), 3) == 0:
            print('transcript does not have the expected format')
        else:
            if '.awd.gz' in file:
                file = file[:-7]
            elif '.gz' in file:
                file = file[:-3]
            file_name = os.path.join(out_path, file +'.txt')
            with open(file_name, "w") as text_file:
                for line in af_trscrpt:
                    text_file.write(line +'\n')
                    
    return

def prep_kaldi(conv_table, in_path, out_path):
# function to load the kaldi made transcripts and convert the phonemes to 
# articulatory featers. Expects transcripts of format starttime \n endtime \n phoneme \n etc
    paths = os.listdir(in_path)
    for file in paths:
        f = os.path.join(in_path, file)   
        with codecs.open(f, 'rb', encoding = 'latin-1') as trans:
            raw_trscrpt = trans.read()
        split_trscrpt = raw_trscrpt.split()
        af_trscrpt = phone2af(split_trscrpt, cgn_dict)
        out = os.path.join(out_path, file)
        with open(out, "w") as text_file:
            for line in af_trscrpt:
                text_file.write(line +'\n')
# create the phoneme to af conversion dictionary
cgn_dict = phoneme_dict(table_loc = "/home/danny/Downloads/Finetracker/Preprocessing/feature_table.txt")
# load and process the transcripts
prep_kaldi(cgn_dict, in_path = "/home/danny/Downloads/Finetracker/split_ali", out_path = '/home/danny/Downloads/Finetracker/transcripts')

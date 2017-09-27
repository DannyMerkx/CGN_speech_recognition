#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 20:26:32 2017

@author: danny
"""

# create a list of all phones we wish to model. You could simply 
# use a known list of phonemes for your database. However with this 
# script you can see which phonemes are actually used in your database. 

from label_func import parse_transcript
from data_functions import list_files
import codecs
datapath="/home/danny/Downloads/nl"
# path to save phones.txt. Your phones should end up in your
# kaldi projects data/local/lang folder
phones_loc="/home/danny/Downloads/Finetracker/kaldi/"
# silence phone, change if you want a different silence phoneme ditto for out of vocab
sil= 'sil'
oov ='<oov>'
pattern= "N[0-9]+_FON"
# list input files and sort (not necesarry but it might make it easier to manually
# look in your files if something goes wrong)
input_files = list_files(datapath)
input_files.sort()
# remove durational info, headers etc.
trans=[parse_transcript(pattern,datapath+"/"+y) for y in input_files]

# extract all phonetic transcribed words
phon_trans=[]
# for each transcript in the list trans
for tr in trans:
    # pick with step size 3 (skip durational info)
    for x in range (2,len(tr),3):
        # split the transcript (awd transcripts should already be split but somehow in CGN some
        # sentences are not properly split yet)
        split_tr = tr[x].replace('_',' ').split()
        # add the words to the word list
        for phone in split_tr:
                phon_trans.append(phone)

# strip punctuation and garbage phones
phon_trans = [x.strip('[]#.?!=\"')for x in phon_trans]

# remove some cgn annotation for foreign and dialect words (e.g. *v and *u which are added to the end of the word)
for x in range (0,len(phon_trans)):
    # simply remove the last two characters
    if len(phon_trans[x])>1 and phon_trans[x][-2] == '*':
        phon_trans[x]=phon_trans[x][:-2]
              
phones=[]
for phon in phon_trans:
    # kaldi want spaces between each phone so we need to split the phonetic 
    #transcripts
    splitphones=''
    for y in range(0,len(phon)):
        # since some phonemes have 2 characters we cannot simply split between
        # each char. The exceptions are hard coded, when using a database with
        # diffent phonemes you might need to add them (this uses +, ~ and :)
        if y == len(phon)-1:
            splitphones=splitphones + phon[y]
        elif phon[y+1]==':' or phon[y+1]=='+' or phon[y+1]=='~':
            splitphones=splitphones + phon[y]
        else:
            splitphones= splitphones + phon[y] + ' '
    split = splitphones.split()
    for s in split:
        phones.append(s)

phones=list(set(phones))
# add the phones to the text file
phon_file = codecs.open(phones_loc+ 'nonsilence_phones.txt', "w")

for phon in phones:
        line  = phon+"\n"
        line.encode('utf-8')
        phon_file.write(line)
phon_file.close()
# add the silence and out of vocab phones to silence_phones.txt
phon_file = codecs.open(phones_loc+ 'silence_phones.txt', "w")
oov.encode('utf-8')
phon_file.write(oov+ '\n')
sil.encode('utf-8')
phon_file.write(sil +'\n')
phon_file.close()
# add the silence phone to optional_silence.txt
phon_file = codecs.open(phones_loc+ 'optional_silence.txt', "w")
phon_file.write(sil)
phon_file.close()      
#note on CGN: even though the official CGN transcription protocol
#lists the symbol 9: for the foreign sound in e.g. the word "freule"
# I found that in the transcription files the symbol Y: is used
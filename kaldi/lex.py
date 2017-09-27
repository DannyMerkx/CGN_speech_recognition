#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:59:31 2017

@author: danny
"""
# Script that reads the automatic phonetranscription from cgn in
# order to create a lexicon for kaldi. Could be used to create a 
# lexicon from scratch or just to complete an existing lexicon.
# since the phonetic transcripts of CGN are not all hand checked,
# there is a small chance of mistakes in there, so it seems best just
# to use this to add unknown words to an existing lexicon
from label_func import parse_transcript
from data_functions import list_files
# takes datapath to your awd files and an oov token
def create_lexicon(datapath, oov):
    # pattern to retrieve the ortographic transcript part
    pattern= "N[0-9]+"
    #pattern to retrieve the phonetic transcript part
    pattern2= "N[0-9]+_FON"
    input_files = list_files(datapath)
    input_files.sort()
    striplist = '._,?!=\"'    
    # remove durational info headers etc.
    word_trans=[parse_transcript(pattern,datapath+"/"+y) for y in input_files]
    phone_trans=[parse_transcript(pattern2,datapath+"/"+y) for y in input_files]  
    phones=[]
    words=[]    
    # extract words and phoneme transcriptions. strip punctuation and convert
    # to lower case.    
    for word, phone in zip(word_trans, phone_trans):
        for x in range (2,len(word),3):
            # replace some special tokens either with words (% with procent) or whitespace. This list MUST be the same as in kaldi_data_train.py
            word[x] = word[x].replace(u'\xb1',u'plusminus').replace(u'\xd7',u'').replace(u'\xb3',u'').replace(u'–',u' ').replace(u'$', u'dollar').replace(u'%',u'procent').replace(u' & ',u' en ').replace(u'&amp',u'en').replace(u'&',u'en').replace(u'\x90',u' ').replace(u'\x91',u' ').replace(u'\x92',u' ').replace(u'\x93',u' ').replace(u'\x94',u' ').replace(u'\x95',u' ').replace(u'\x96',u' ').replace(u'\x97',u' ').replace(u'\x98',u' ').replace(u'\x99',u' ').replace(u'\xbd',u'').replace(u'\xff',u'').replace(u'\u2663',u'').replace(u'\u2666',u'').replace(u'\u2660',u'').replace(u'\u2665',u'').replace(u'\xb9',u'').replace(u'\xb2',u'').replace(u'\u2070',u'').replace(u'\u2079',u'').replace(u'\u2074',u'').replace(u'\u0660',u'').replace(u'\u2075',u'').replace(u'\u2071',u'').replace(u'\u2072',u'').replace(u'\u2073',u'').replace(u'\u2076',u'').replace(u'\u2077',u'').replace(u'\u2078',u'').replace(u'\u2792',u'').replace(u'\u2082',u'').replace(u"1/2","half").replace(u"/",u" ").replace(u'~',u'')
            word[x] = word[x].split()
            phone[x] =phone[x].split()
            # if words and phones are of same length just append to list
            # otherwise split the phoneme transcript (some _ are not properly)
            # separeted in cgn it seems.
            if len (word[x]) != len(phone[x]):               
                phone[x]= [p for q in phone[x] for p in q.split('_')]                 
            for z in range(0,len(word[x])):
                words.append(str.lower(word[x][z].strip(striplist)))
                # do NOT lower case the phonemes, A and a are different phonemes
                phones.append(phone[x][z].strip(striplist))
    # remove silence.and underscore (underscore is used when 2 adjacent words share
    # a phoneme and it is unclear where the boundary is)
    phones = [x for x in phones if len(x)>0 and not x =='_']
    words = [x for x in words if len(x)>0 and not x =='_']          
    # remove * annotations of cgn (e.g. *v for foreign words)
    for x in range(0,len(words)):
        if '*' in words[x]:
            words[x] = words[x][:-2]    
    # create a lexicon, that is tuples of words and their pronunciation in phonemes   
    lexicon=[]
    for x in range (0,len(phones)):
        # kaldi want spaces between each phone so we need to split the phonetic 
        #transcripts
        splitphones=''
        for y in range(0,len(phones[x])):
            # since some phonemes have 2 characters we cannot simply split between
            # each char. The exceptions are hard coded, when using a database with
            # diffent phonemes you might need to add them (this uses +, ~ and :)
            if y == len(phones[x])-1: # do not check x+1 for the last in list as it creates errors
                splitphones=splitphones + phones[x][y]
            elif phones[x][y+1]==':' or phones[x][y+1]=='+' or phones[x][y+1]=='~':
                splitphones=splitphones + phones[x][y]
            else:
                splitphones= splitphones + phones[x][y] + ' '
        # ggg xxx and Xxx are used to transcribe unintelligeble parts. We replace these
        # with an out of vocab token. again hardcoded, you might need to add your own exceptions here
        if not 'ggg' in words[x] and not 'xxx' in words[x] and not 'Xxx' in words[x]:        
            lexicon.append((words[x],splitphones))
        else:
            lexicon.append((oov[0],oov[1]))
    # remove doubles
    lexicon=list(set(lexicon))
    # put the lexicon in dictionary form to map multiple pronunciations
    # onto one ortographic form
    ref={}
    for x in lexicon:       
        try:
            ref [x[0]].append(x[1])
        except:
            ref[x[0]] = list()
            ref[x[0]].append(x[1])  
    return(ref)

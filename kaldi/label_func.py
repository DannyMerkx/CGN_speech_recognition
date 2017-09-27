#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 14:21:45 2016

@author: danny
"""
import gzip
import re
import numpy as np
import math
import codecs
# functions for cleaning the transcript and labelling the data

def cleanup (text,pattern):
    #cleans up the raw text file
    match= pattern.search(text)
    begin=match.span()[0]
    cl_text=text[begin:-1]
    return cl_text
    
def parse_transcript (pattern, loc):
# function to parse the CGN transcripts, requires a pattern. CGN transcripts like the awd transcripts contain the ortographic, phonetic and segmented phonetic transcripts. Use  "N[0-9]+" , "N[0-9]+FON" or "N[0-9]+_SEG" to retrieve these parts respectively
  
    # parse the raw transcript
    regex= re.compile(pattern)
    # works with zipped files as CGN provides them or unzipped files
    try:
        with gzip.open(loc,'rb') as file:
            x=file.read().decode('latin-1')
    except:
        with codecs.open(loc,'rb') as file:
            x=file.read().decode('latin-1')

    # cleanup removes the parts of the transcript we dont need 
    cleaned_trscrpt = cleanup(x,regex)
    # split the transcript in seperate lines
    split_trscrpt = cleaned_trscrpt.splitlines()
   
    # the 4th line of each transcript should say how many segments there are, multiply by three because each segment consists of 3 lines.
    trscpt_size = int(split_trscrpt[3])*3
    # at last remove some empty lines and header lines so only the phonemes and their
    # start and end time are left
    final_trscrpt= split_trscrpt[4:trscpt_size+4]  
    return (final_trscrpt)


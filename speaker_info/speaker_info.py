#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:25:11 2017

@author: danny
"""

import glob
import os
import argparse
import gzip
import re
import sklearn.model_selection
import numpy

# open the file with CGN speaker info.
cgn_info = []
with open('/home/danny/Downloads/speakers.txt') as file:
    for line in file:
        cgn_info.append(line.split())

# open the transcripts for the desired CGN component (to filter only those speakers
# appearing in the component)
awd = glob.glob(os.path.join('/home/danny/Downloads/comp-o/nl', '*awd.gz' ))

# extract speaker id from the transcripts (use 'N[0-9]+' for the belgian part of CGN)
regex = re.compile('N[0-9]+')

comp_ids = []
file_ids = []
for file in awd:
   with gzip.open(file, 'rb') as f:
    file_content = f.read().decode('latin-1') 
    span = re.search(regex, file_content).span()
    comp_ids.append(file_content[span[0]:span[1]])
    base = os.path.basename(file)
    file_ids.append([base[:-7], file_content[span[0]:span[1]] ])

# some speakers appear in multiple files, take the set of speakers
speaker_set = list(set(comp_ids))

# retrieve the speaker info from the cgn info file for the speakers in our component
inf = [numpy.array(inf) for inf in cgn_info if inf[4] in comp_ids]

# get the number of files each speaker appears in 
speaker_count = [[x, comp_ids.count(x)] for x in set(comp_ids)]
speaker_count.sort()
# put the info in a list
speaker_info = []
# keep track of speaker ages so we can bin them later
age = []
for x in inf:
    # this takes sex, age (date of cgn - speaker birthyear), speaker first language and education
    # age unknown is written as 19xx in CGN, so we cannot derive age (map to 999).
    if x[6] != '19xx':
        age.append(2004 - int(x[6]))
        speaker_info.append([x[4], x[5], 2004 - int(x[6]),x[9],x[19],x[21]])
    else:
        speaker_info.append([x[4], x[5], 999, x[9], x[19]])

# map age to bins based on percentile   
for x in speaker_info:
    if x[2] == 999:
        continue
    elif x[2] <= numpy.percentile(age, 25):
        x[2] = 1
    elif x[2] <= numpy.percentile(age, 50):
        x[2] = 2
    elif x[2] <= numpy.percentile(age, 75):
        x[2] = 3
    elif x[2] <= numpy.percentile(age, 100):
        x[2] = 4

# add the number of files each speaker appears in           
for x in range(len(speaker_count)):
    speaker_info[x].append(speaker_count[x][1])

ids = []
info = []
for x in range(len(speaker_info)):
    temp = speaker_info[x][1:]
    ids.append(speaker_info[x][0])
    info.append(temp)

info = numpy.array(info)

mapping_sex, info[:, 0] = numpy.unique(info[:,0], return_inverse=True)
mapping_age, info[:, 1] = numpy.unique(info[:,1], return_inverse=True)
mapping_lang, info[:, 2] = numpy.unique(info[:,2], return_inverse=True)
mapping_education, info[:,3] = numpy.unique(info[:,3], return_inverse=True)
mapping_occupation, info[:,4] = numpy.unique(info[:,4], return_inverse=True)
mapping_count, info[:, 5] = numpy.unique(info[:,5], return_inverse=True)

mapping_speaker, speaker_set = numpy.unique(ids, return_inverse=True)

x =False
while x==False:
# split the dataset into train test and val sets with simmilar statistics concerning the attributes 
# extracted above.
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    sss.get_n_splits(speaker_set, info[:,0])

    for train_index, test_index in sss.split(speaker_set, info[:,0]):
        X_train, X_test = speaker_set[train_index], speaker_set[test_index]
        y_train, y_test = info[train_index], info[test_index]

    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1)
    sss.get_n_splits(X_test, y_test[:,0])

    for val_index, test_index in sss.split(X_test, y_test[:,0]):
        X_val, X_test = X_test[val_index], X_test[test_index]
        y_val, y_test = y_test[val_index], y_test[test_index]

    a=y_train[:,1].astype(int).mean()
    b=y_test[:,1].astype(int).mean()
    c=y_val[:,1].astype(int).mean()

    if abs(a-b) < 0.1 and abs(a-c) <0.1 and abs(b-c) <0.03:

        a=y_train[:,3].astype(int).mean()
        b=y_test[:,3].astype(int).mean()
        c=y_val[:,3].astype(int).mean()
        if abs(a-b) < 0.1 and abs(a-c) <0.1 and abs(b-c) <0.03:
            a=y_train[:,4].astype(int).mean()
            b=y_test[:,4].astype(int).mean()
            c=y_val[:,4].astype(int).mean()
            if abs(a-b) < 0.1 and abs(a-c) <0.1 and abs(b-c) <0.03:
                x =True


train = [[ids[x],'train'] for x in X_train]
test = [[ids[x],'test'] for x in X_test]
val = [[ids[x],'val'] for x in X_val]

split = numpy.concatenate([numpy.array(train),numpy.array(test),numpy.array(val)])

file_split = []
for x in file_ids:
    for y in split:
        if x[1] == y[0]:
            file_split.append([x[0], y[1]])

numpy.save('/home/danny/Downloads/split.npy', file_split)
numpy.save('/home/danny/Downloads/test.npy', y_test)
numpy.save('/home/danny/Downloads/val.npy', y_val)
numpy.save('/home/danny/Downloads/train.npy', y_train)
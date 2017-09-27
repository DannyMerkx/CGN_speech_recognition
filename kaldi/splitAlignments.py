#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/bin/sh

#  splitAlignments.py
#  
#
#  Created by Eleanor Chodroff on 3/25/15.
#
#
#
import sys,csv, codecs
results=[]

#name = name of first text file in final_ali.txt
#name_fin = name of final text file in final_ali.txt

name = "fn001500"

file_loc = "/scratch/danny/kaldi/egs/myexp/exp/tri4a_ali/final_ali.txt"
# don't forget to create a split_ali folder in tri4a_ali
new_file_loc = "/scratch/danny/kaldi/egs/myexp/exp/tri4a_ali/split_ali/"
try:
    with open(file_loc) as f:
        next(f) #skip header
        for line in f:
            columns=line.split("\t")
            name_prev = name
            name = columns[1][7:]
            if (name_prev != name):
                try:
                    fwrite = codecs.open(new_file_loc + (name_prev)+".txt",'w')
                    writer = csv.writer(fwrite)
                    fwrite.write("\n".join(results))
                    fwrite.close()
                #print name
                except:
                    print ("Failed to write file")
                    sys.exit(2)
                del results[:]
                results.append(line[0:-1])
            else:
                results.append(line[0:-1])
except:
    print ("Failed to read file")
    sys.exit(1)
# this prints out the last textfile (nothing following it to compare with)
try:
    fwrite = codecs.open(new_file_loc + (name_prev)+".txt",'w')
    writer = csv.writer(fwrite)
    fwrite.write("\n".join(results))
    fwrite.close()
                #print name
except:
    print ("Failed to write file")
    sys.exit(2)
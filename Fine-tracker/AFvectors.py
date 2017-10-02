#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:19:51 2017

@author: danny
"""

import os
import numpy

def load_predictions(loc, file):
    # load duration prediction
    preds_file = numpy.load(os.path.join(loc,file))

    preds =[]

    for x in preds_file:
        preds.append(preds_file[x])
    # get the shape
    shape = numpy.shape(preds)
    # reshape 5d output matrix to 2d 
    preds = numpy.reshape(numpy.transpose(preds),[shape[4],shape[1]*shape[3]])    

    return preds
def load_targets(loc, file):
    
    targs =[]
    targs_file = numpy.load(os.path.join(loc, file))
    for x in targs_file:
        targs.append(targs_file[x])

    shape = numpy.shape(targs)

    targs = numpy.reshape(numpy.transpose(targs),[shape[2]* shape[1]])

    return targs

def reorder(loc, class_dict, vector_order, af_order):      
    # reorder the AFs given some vector ordering needs a dictionary with all
    # the afs and the labels and a desired order for classes and features and the
    # location of the prediction files
    
    # list all the prediction files    
    files = os.listdir(loc)
    files.sort()
    prediction_files = [f for f in files if 'predictions' in f]
    
    
    af_vectors = {}
    # load and reorder the features
    for af in class_dict:     
        # load the predictions
        af_vectors[af] =  load_predictions(loc, [x for x in prediction_files if af in x][0])
        # reorder them in the order required by fine-tracker
        order = []
        # for all labels in the current AF look up the order in which they should appear
        for x in class_dict[af]:
            order.append(vector_order[x])
        # reorder the features
        af_vectors[af]=af_vectors[af][numpy.argsort(order)]
    
    articulatory_features= []
    for x in af_order:
        for y in af_vectors[x]:
            articulatory_features.append(y)
    return articulatory_features        
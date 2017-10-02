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

#!/usr/bin/env python
from __future__ import print_function

import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import tables
from dnn_data import Split_dataset
import argparse
# this scripts implements a deep convolutional neural network for frequency spectral features. The first layer applies the triangular filterbanks to the frequency spectral features, the input to the rest of the layers are than equal to the filterbank networks. The filterbank weights are a trainable parameter though.
parser = argparse.ArgumentParser(description='Create and run an articulatory feature classification DNN')
parser.add_argument('-load_weights', type = bool, default = False, 
                    help = 'load a pre-trained model (True), or initialise a new one (False), default: False')
parser.add_argument('-weight_loc', type = str, default = './model.npz',
                    help = 'location of pretrained weights, default: ./model.npz ')
parser.add_argument('-data_loc', type = str, default = '/data/processed/freq_spectrum.h5',
                    help = 'location of the feature file, default: /data/processed/freq_spectrum.h5')
parser.add_argument('-batch_size', type = int, default = 512, help = 'batch size, default: 512')
parser.add_argument('-splice_size', type = int, default = 5, help = 'splice size, default: 5')
parser.add_argument('-test', type = bool, default = False, 
                    help = 'set to True to skip training and only run the network on the test set, use in combination with a pretrained model of course, default: False')
parser.add_argument('-af', type = str, default = 'manner', 
                    help = 'the articulatory feature you want the network to be trained for (manner, place, voice, frback, height, round, dur_diphthongue or phones)')
parser.add_argument('-feat_type', type = str, default = 'freq_spectrum', 
                    help = 'type of input feature, either mfcc, fbanks, freq_spectrum or raw, default: freq_spectrum')
parser.add_argument('-dropout', type = float, default =0.2,
                    help = 'dropout, probability of setting channels to 0, default: 0.2')
parser.add_argument('-filter_loc', type = str, default = './filters.npy', 
                    help = 'location of the filterbank filters, default: ./filters.npy')
parser.add_argument('-index_loc', type = str, default = '/data/Finetracker/DNN',
                    help = 'location to save/load the index files')
parser.add_argument('-split_loc', type = str, default = '/data/Finetracker/DNN/split.npy',
                    help = 'location of the npy file with the split information')
parser.add_argument('-data_size', type = float, default = 1.0, help = 'size of the data to use between 0 and 1, default: 1 (full dataset)')
args = parser.parse_args()


# dictionary of articulatory features with the number of classes and the idx of the labels in the feature file
af_dict = {'phones': [0, 38], 'manner': [1, 8], 'place': [2, 8], 'voice': [3, 2], 'frback': [4, 4],
           'height': [5, 4], 'round': [6, 3], 'dur_diphthongue': [7, 4]}
out_units = af_dict[args.af][1]
l_idx = af_dict[args.af][0]
# dictionary of some dimension presets for different feature types
feature_dict = {'mfcc': 39, 'fbanks': 64, 'freq_spectrum': 257, 'raw': 400}
input_size = feature_dict[args.feat_type]
# open the data file
data_file = tables.open_file(args.data_loc, mode='r+') 
#get a list of feature nodes
f_nodes = data_file.root.features._f_list_nodes()
#get a list of label nodes
l_nodes = data_file.root.labels._f_list_nodes()
# total number of nodes (i.e. files) 
n_nodes= len(f_nodes)
# load the filterbank filters
filters = np.load(args.filter_loc)
filters = np.reshape(filters,[64, 1, 1, 257]).astype('float32')
# load dataset indices
print('creating train, val and test sets')
[Train_index, Val_index, Test_index] = Split_dataset(f_nodes, l_nodes, args.splice_size, args.index_loc, args.split_loc, args.data_size)  

print('DNN training settings: input features ' + args.feat_type + '\n'
+ 'articulatory feature ' + args.af + '\n' + 'dropout' + str(args.dropout))

# layer that returns the log of its input
class LogLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        # clip the tensor at a value close to 0 to prevent taking log(0)
        return T.log(T.clip(input,1e-24,10000))
# layer to transpose its input
class TransposeLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        return np.transpose(input,(0,1,3,2))
        
def build_dnn(input_var=None):
    # input layer
    network = lasagne.layers.InputLayer(shape = (None, 1, (args.splice_size*2)+1, input_size),
                                        input_var = input_var)
    # conv layer applying the filters
    network = lasagne.layers.Conv2DLayer(network, num_filters = np.shape(filters)[0], filter_size = (1,np.shape(filters)[3]), stride = (1,1), pad = 'valid', W = filters, flip_filters = False)

    # the reshape layer cannot cast our input into the right shape because lasagne only supports C rather than fortran style reshape order. Instead we switch the last 2 dimensions and perform a transpose operation followed by another reshape to get the values in the right order
    network = lasagne.layers.ReshapeLayer(network,shape=([0],[3],[1],[2]))
    network = TransposeLayer(network)
    network = lasagne.layers.ReshapeLayer(network,shape=([0],[1],(args.splice_size*2)+1,np.shape(filters)[0]))
    
    # take the log fbank features
    network = LogLayer(network)     
    
    # block 1   
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(network, num_filters = 64, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 64, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network,pool_size = (1, 2), stride = (1, 2), ignore_border = True)
   
    # block 2
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 128, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 128, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (1, 2), stride = (1, 2), ignore_border = True)
    
    # block 3
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 128, filter_size = (3, 3), stride=(1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 128, filter_size = (3, 3), stride=(1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network, pad = (1, 0), pool_size = (2, 2), stride = (2, 2), ignore_border = True)

    # block 4
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 256, filter_size = (3, 3), stride=(1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 256, filter_size = (3, 3), stride=(1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network, pad = (1, 0), pool_size = (2, 2), stride = (2, 2), ignore_border = True)

    # block 5
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 256, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.Conv2DLayer(lasagne.layers.dropout_channels(network, p = args.dropout), num_filters = 256, filter_size = (3, 3), stride = (1, 1), pad = 'same', W = lasagne.init.GlorotUniform(), nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.MaxPool2DLayer(network, pool_size = (2, 2),stride=(2,2),ignore_border = True)

    # fully connected
    network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout_channels(network, p = args.dropout),
            num_units = 2048,
            nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p = args.dropout),
            num_units = 2048,
            nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p = args.dropout),
            num_units = 2048,
            nonlinearity = lasagne.nonlinearities.rectify))
    network = lasagne.layers.batch_norm(lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p = args.dropout),
            num_units = 2048,
            nonlinearity = lasagne.nonlinearities.rectify))
    #output
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p = args.dropout),
            num_units = out_units,
            nonlinearity = lasagne.nonlinearities.softmax)

    return network

#iterate minibatches where data is loaded to memory at every iteration. 
def iterate_minibatches(index, batchsize, splice_size, shuffle=True):  
    if shuffle:
        np.random.shuffle(index)
    for start_idx in range(0, len(index) - batchsize + 1, batchsize):               
        excerpt = index[start_idx:start_idx + batchsize]        
        inputs=[]
        targets=[]
        for ex in excerpt:
            # retrieve the frame indicated by index and splice it with surrounding frames
            inputs.append([f_nodes[ex[1]][ex[2]+x] for x in range (-splice_size,splice_size+1)])
            targets.append(l_nodes[ex[1]][ex[2]][l_idx])
        shape= np.shape(inputs)
        inputs = np.float32(np.reshape(inputs,(shape[0],1,shape[1],shape[2])))
        targets = np.uint8(targets)
        yield inputs, targets

# ############################## Main program ################################
def main(num_epochs = 5):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_dnn(input_var)   
    # load existing parameters if applicable
    if args.load_weights:   
        with np.load(args.weight_loc) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            param_values.insert(0, filters)
        lasagne.layers.set_all_param_values(network, param_values)
    # Create a loss expression for training
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # Create update expressions for training.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    # Create a loss expression for validation/testing.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch 
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)
    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    
    # function which returns the networks' predictions
    out_fn = theano.function([input_var], [test_prediction])

    # Finally, launch the training loop.
    print("Starting training...")
    val_acc=1
    prev_val_acc=0
    epoch=0
# train while val accuracy is rising and number of epochs is not reached. Set args.test to True to only test on an existing model and skip training. 
    while val_acc > prev_val_acc and epoch < num_epochs and args.test == False:
    ################################################################
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(Train_index, args.batch_size, args.splice_size, shuffle = False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
# save the model parameters
        np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
        # And a full pass over the validation data:
        prev_val_acc=val_acc
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(Val_index, args.batch_size, args.splice_size, shuffle = False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        epoch=epoch+1
    # After training, we compute and print the test error:
    print ('computing test accuracy')
    test_err = 0
    test_acc = 0
    test_batches = 0
        # collect and save the predictions and true labels for analysis
    targs = []
    preds = []
    for batch in iterate_minibatches(Test_index, args.batch_size, args.splice_size, shuffle = False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        output = out_fn(inputs)   
        test_err += err
        test_acc += acc
        test_batches += 1
        
        targs.append(targets)
        preds.append(output)
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))
    #save the predictions targets and model weights
    np.savez('predictions_' + args.af + '.npz', preds)
    np.savez('targets_' + args.af + '.npz', targs)
    np.savez(args.af + '_model.npz', *lasagne.layers.get_all_param_values(network))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    main()

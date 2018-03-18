from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from collections import defaultdict
import operator
import pdb

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    #parameter of k nearest
   
    k = 5
    dict_ = defaultdict(int)
    distance_feats = distance.cdist(test_image_feats, train_image_feats)
    test_predicts = []
    k_idx_lists = np.argsort(distance_feats)[:,:k]
    for k_idx_list in k_idx_lists:
        labels = [ train_labels[idx] for idx in k_idx_list]
        for label in labels:
            dict_[label] += 1
        test_predicts.append(max(dict_.items(), key=operator.itemgetter(1))[0])
        dict_.clear()
    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts

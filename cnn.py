#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g

    def __init__(self, max_word_size, char_embed_size, num_filter, kernel_size, padding_size):
    	'''
    	Init high way initiation
        @param max_word_size: size of the max word length in the corpus
        @param char_embed_size: size of the character embedding
        @param num_filter: number of filters (f)
        @param kernel_size: length of each kernels in CNN
    	'''
    	super(CNN, self).__init__()
    	self.max_word_size = max_word_size
    	self.char_embed_size = char_embed_size
    	self.num_filter = num_filter
    	self.kernel_size = kernel_size
    	# in_channels is the char level embed size, on each row, how many value when CNN
    	# out_channels: number of filters, controls how many output
    	self.conv1 = nn.Conv1d(in_channels = self.char_embed_size,
    		out_channels = self.num_filter,
    		kernel_size = self.kernel_size,
    		stride=1, 
    		padding = padding_size, 
    		dilation=1, 
    		bias=True, )

    	# Not use the Maxpool1
    	# self.maxpool1 = nn.MaxPool1(kernel_size = (self.max_word_size-self.kernel_size+1+padding_size*2),
    	# 	stride=None, 
    	# 	padding=0, 
    	# 	dilation=1, 
    	# 	return_indices=False, 
    	# 	ceil_mode=False)

    def forward(self, x_reshape:torch.Tensor):
	    ''' Apply CNN on x_reshape
	    @param x_reshape: tensor of (batch_size, char_embed_size, max_word_size)


	    @return x_conv_out: tensor of (batch_size, num_filter) 
	    '''
	    # print("input shape", x_reshape.size())
	    x_conv = self.conv1(x_reshape) # (batch_size, number of filters, max_word_size -k + 1)
	    # print("x_conv size-----------------", x_conv.size())
	    # x_conv_out = self.maxpool1(F.relu(x_conv)).squeeze() # (batch_size, number of filters)
	    x_conv_out = torch.max(F.relu(x_conv),2)[0] # tuple of (value, index), only take value
	    # print("x_conv_out size-----------------",type(x_conv_out))
	    # print("x_conv_out squeeze size-----------------",x_conv_out.squeeze().size())

	    return x_conv_out


    ### END YOUR CODE


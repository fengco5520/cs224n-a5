#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    def __init__(self, embed_size,dropout_rate = 0.5):
    	'''
    	Init high way initiation
        @param embed_size: size of the word embedding
        @param dropout_rate: dropout rate
    	'''
    	super(Highway, self).__init__() # init the highway class with inherit from nn.Module
    	self.embed_size = embed_size
    	self.dropout_rate = dropout_rate

    	self.projection = nn.Linear(self.embed_size,self.embed_size,bias = True) # projection layer
    	self.gate = nn.Linear(self.embed_size, self.embed_size, bias = True) #gate layer operation
    	self.dropout = nn.Dropout(p=dropout_rate) # dropout layer


    def forward(self, x_conv_out: torch.Tensor):
    	""" Apply convolution layer to highway network

        @param x_conv_out: tensor of (batch_size, embed_size)

        @returns dropout(x_conv_highway): tensor of (batch_size, embed_size)
        """

    	x_conv_projection = F.relu(self.projection(x_conv_out))
    	x_conv_gate = torch.sigmoid(self.gate(x_conv_out))
    	x_conv_highway =  torch.mul(x_conv_projection,x_conv_gate) + torch.mul((1-x_conv_gate),x_conv_out)
    	return self.dropout(x_conv_highway)

    ### END YOUR CODE


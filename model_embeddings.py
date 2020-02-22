#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    Convert the batch of sents of words into char level embedding
    Feed the char level embedding to CNN/Highway to generate the x_conv_out
    """
    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>'] # notice that in assignment 4 vocab is of type (Vocab), not (VocabEntry) as assignment 5.
        # self.embeddings = nn.Embedding(len(vocab.src), word_embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 
        self.word_embed_size = word_embed_size # word_embed_size != max_word_length
        self.vocab = vocab
        self.char_embed_size = 50
        self.dropout_rate = 0.3
        self.kernel_size = 5
        self.max_word_length = 21


        pad_token_idx = vocab.char2id['∏']
        # tensor(total_num_char, embedding_dim) initial random generated 
        # nn.parameters is trainable
        # nn.Embedding is a trainablable parameters
        self.embeddings = nn.Embedding(num_embeddings = len(vocab.char2id),
            embedding_dim =self.char_embed_size , padding_idx=pad_token_idx)

        # init a CNN layer
        self.cnn = CNN(max_word_size = self.max_word_length, 
            char_embed_size=self.char_embed_size, 
            num_filter = self.word_embed_size, 
            kernel_size = self.kernel_size, 
            padding_size = 1)


        self.highway = Highway(embed_size = self.word_embed_size, dropout_rate = self.dropout_rate)

        
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1h
        # input is the  batch * sentences (num of words in the sent) * word_len (number of letter in a word)
        x_char = self.embeddings(input) # (sent_length, batch_size, max_word_length, char_embed_size)
        # print("Input size", input.size())
        # print("Char level size",x_char.size())
        x_reshape =  x_char.permute(0,1,3,2) #(sent_length, batch_size, char_embed_size, max_word_length)
        # print("word size ",  self.word_embed_size)
        # print("reshaped Char level size", x_reshape.size())

        sent_length, batch_size, _,max_word_length =  x_reshape.size()

        # Conv1d only could take (num of words, char_embed_size, word_length)
        # Does it matter how to reorder the data? or initialization, not matter????
        x_reshape_deep =x_reshape.reshape(sent_length*batch_size, self.char_embed_size,max_word_length)  #(sent_length * batch_size, char_embed_size, max_word_length)
        # print("reshaped deepen Char level size", x_reshape_deep.size())

        # reshape(6,2) sequentially change the shape

        x_conv_out = self.cnn(x_reshape_deep)  #(sent_length* batch_size, num_filters) num_filter != max_word_length
        # print("x_conv_out size ", x_conv_out.size())

        x_highway = self.highway(x_conv_out)  #(sent_length* batch_size, num_filters) 
        # print("x_highway size", x_highway.size())

        return x_highway.reshape(sent_length,batch_size,self.word_embed_size)

        ### END YOUR CODE


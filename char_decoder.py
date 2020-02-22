#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        super(CharDecoder, self).__init__()
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(self.target_vocab.char2id)) # st
        self.decoderCharEmb = nn.Embedding(len(self.target_vocab.char2id), char_embedding_size, padding_idx=self.target_vocab.char_pad)


    def forward(self, inputs, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input (Tensor): tensor of integers, shape (length, batch_size), ids
        @param dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores (Tensor): called s_t in the PDF, shape (length, batch_size, self.vocab_size)
        @returns dec_hidden (tuple(Tensor, Tensor)): internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Implement the forward pass of the character decoder.
        char_embedding = self.decoderCharEmb(inputs) # dictionary to keep the embedding
        hidden_states, dec_hidden = self.charDecoder(char_embedding, dec_hidden) # h_t: (length, batch_size, hidden_size)
        #output: (sequence_length, batch_size, hidden_size), it contains all the previous sequence's h_t
        # ht is only the last step's ht 
        scores = self.char_output_projection(hidden_states) #st:(length, batch_size, self.vocab_size)

        return scores, dec_hidden
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence (Tensor): tensor of integers, shape (length, batch_size). Note that "length" here and in forward() need not be the same.
        @param dec_hidden (tuple(Tensor, Tensor)): initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch_size, hidden_size)

        @returns The cross-entropy loss (Tensor), computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss. Check vocab.py to find the padding token's index.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} (e.g., <START>,m,u,s,i,c,<END>). Read the handout about how to construct input and target sequence of CharDecoderLSTM.
        ###       - Carefully read the documentation for nn.CrossEntropyLoss and our handout to see what this criterion have already included:
        ###             https://pytorch.org/docs/stable/nn.html#crossentropyloss
        
        #char_sequence is the whole  sequence [<Start>, m,u,s,i,c,<END>]
        inputs = char_sequence[:-1] # remove the last item from the sequence
        scores, _ = self.forward(inputs, dec_hidden) #score (length-1, batch_size, self.vocab_size)

        scores = scores.contiguous().view(-1, scores.shape[-1]) # (length-1, batch_size, self.vocab_size) -> ((length-1)* batch_size, self.vocab_size)

        targets = char_sequence[1:].contiguous().view(-1) #(length-1, batch_size) -> (length-1*batch_size)
        
        loss = nn.CrossEntropyLoss(reduction = 'sum' , ignore_index = self.target_vocab.char2id['∏'])
        # loss(scores, target) scalar single value, sum all the values
        return loss(scores,targets)
        
        ### END YOUR CODE



    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates (tuple(Tensor, Tensor)): initial internal state of the LSTM, a tuple of two tensors of size (1, batch_size, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length (int): maximum length of words to decode

        @returns decodedWords (List[str]): a list (of length batch_size) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2c
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use initialStates to get batch_size.
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - You may find torch.argmax or torch.argmax useful
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        batch_size = initialStates[0].size(1)
        dec_hidden = initialStates
        start_char = [[self.target_vocab.start_of_word]]*batch_size # [[1],[1],...]
        inputs = torch.tensor(start_char, dtype = torch.long, device = device).t() #(batch_size, 1) -> (1, batch_size)
        # self.forward input:(length, batch_size)
        decodeChar = [] #decodeChar: list of max_length of (1, batch_size)

        for step in range(max_length):
            score, dec_hidden = self.forward(inputs, dec_hidden) # score :(length =1, batch_size, self.vocab_size)
            inputs = score.argmax(2) #input:(length =1, batch_size) index of the max char
            decodeChar.append(inputs) # id of max score char 

        decodeChar = torch.stack(decodeChar, dim = 0) # (max_length, 1, batch)
        decodeChar = decodeChar.permute(2,1,0).squeeze(1).tolist() # (batch, max_length) 
        #squeeze(1) only remove the 1th dimension. if batch ==1 (1,1,21), squeeze() will also remove batch size 

        output = []
        # print("DEcoder ===========",decodeChar)

        for charids in decodeChar:
            # print("TEST ===========",charids)
            word = ''

            for charid in charids:
                char = self.target_vocab.id2char[charid]

                if char == '}':
                    break
                if char !='∏':
                    word+=char
            output.append(word)
        return output
















        ### END YOUR CODE


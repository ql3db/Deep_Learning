import random

""" 			  		 			     			  	   		   	  			  	
Seq2Seq model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import torch
import torch.nn as nn
import torch.optim as optim


# import custom models


class Seq2Seq(nn.Module):
    """ The Sequence to Sequence model.
        You will need to complete the init function and the forward function.
    """

    def __init__(self, encoder, decoder, device, attention=False):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.attention=attention  #if True attention is implemented
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the Seq2Seq model. You should use .to(device) to make sure  #
        #    that the models are on the same device (CPU/GPU). This should take no  #
        #    more than 2 lines of code.                                             #
        #############################################################################
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, source, out_seq_len=None):
        """ The forward pass of the Seq2Seq model.
            Args:
                source (tensor): sequences in source language of shape (batch_size, seq_len)
                out_seq_len (int): the maximum length of the output sequence. If None, the length is determined by the input sequences.
        """

        batch_size = source.shape[0]
        if out_seq_len is None:
            out_seq_len = source.shape[1]
        #############################################################################
        # TODO:                                                                     #
        #   Implement the forward pass of the Seq2Seq model. Please refer to the    #
        #   following steps:                                                        #
        #       1) Get the last hidden representation from the encoder. Use it as   #
        #          the first hidden state of the decoder                            #
        #       2) The first input for the decoder should be the <sos> token, which #
        #          is the first in the source sequence.                             #
        #       3) Feed this first input and hidden state into the decoder          #  
        #          one step at a time in the sequence, adding the output to the     #
        #          final outputs.                                                   #
        #       4) Update the input and hidden weights being fed into the decoder   #
        #          at each time step. The decoder output at the previous time step  # 
        #          will have to be manipulated before being fed in as the decoder   #
        #          input at the next time step.                                     #
        #############################################################################
        # 1) Pass source through the encoder
        # print("test", self.encoder.model_type)
        encoder_outputs, encoder_hidden = self.encoder(source)

        # 2) Initialize input to the decoder as a placeholder token (e.g., zeros)
        decoder_input = source[:, 0:1]
    
        # Initialize the initial hidden state of the decoder with the encoder's final hidden state
        decoder_hidden = encoder_hidden
    
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, out_seq_len, self.decoder.output_size, device=self.device)        
    
        # 3) Loop over sequence length or out_seq_len
        for t in range(out_seq_len):
            # print(t)
            # 4) Feed input and hidden state to the decoder
            # print("init", decoder_input)
            if self.attention == False:
                # print("debug:", decoder_input.size(), decoder_hidden.size())
                # print("here!")
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # print("finally pass this")
            else: 
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs=encoder_outputs, attention=self.attention)
            # print(decoder_output.size())
            # print(decoder_hidden.size())
    
            # Store the decoder output in the outputs tensor
            outputs[:, t, :] = decoder_output.squeeze(1)
            # print("output",outputs)
    
            # Update input for the next time step (use the decoder's prediction as the next input)
            decoder_input = decoder_output.argmax(1).unsqueeze(1)
            # print("translate input",decoder_input)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outputs

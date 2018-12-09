import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import create_dir_if_not_exists
import matplotlib.pyplot as plt


class Seq2Seq(nn.Module):
    def __init__(self, input_size=1, output_size=1, decoder_feat_size=0, hidden_size=8,
                 input_length=512, output_length=128,
                 teacher_forcing=False):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_length = input_length
        self.output_length = output_length
        self.teacher_forcing = teacher_forcing
        self.decoder_feat_size = decoder_feat_size

        self.decoder_input_size = self.output_size + self.decoder_feat_size

        self.encoder_cell = nn.GRUCell(input_size=self.input_size, hidden_size=self.hidden_size)
        self.decoder_cell = nn.GRUCell(input_size=self.decoder_input_size, hidden_size=self.hidden_size)
        self.decoder_fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, decoder_feat=None, y=None):
        """
        :param x: Tensor with shape [input_length, batch_size, input_size]
        :param decoder_feat: Tensor with shape [output_length, batch_size, decoder_feat_size]
        :param y: Tensor with shape [output_length, batch_size, output_size]
        :return: Tensor with shape [output_length, batch_size, output_size]
        """
        if self.teacher_forcing and y is None:
            raise ValueError('y must be provided when using teacher_forcing.')
        if self.decoder_feat_size > 0 and decoder_feat is None:
            raise ValueError('decoder_feat must be provided when decoder_feat_size is not 0.')
        batch_size = x.size(1)
        # encoder
        encoder_hidden = torch.zeros((batch_size, self.hidden_size))
        for t in range(self.input_length):
            encoder_hidden = self.encoder_cell(x[t], encoder_hidden)
        # decoder
        outs = []
        decoder_input = x[-1]

        if decoder_feat is not None:
            decoder_input = torch.stack([x[-1], decoder_feat[0]])

        decoder_hidden = encoder_hidden

        for t in range(self.output_length):
            decoder_hidden = self.decoder_cell(decoder_input, decoder_hidden)
            out_t = self.decoder_fc(decoder_hidden)

            if self.teacher_forcing:
                decoder_input = y[t]
            else:
                decoder_input = out_t

            if decoder_feat is not None and t < self.output_length:
                decoder_input = torch.stack([decoder_input, decoder_feat[t+1]])

            outs.append(out_t)

        outs = torch.stack(outs)
        return outs

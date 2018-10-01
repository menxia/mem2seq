import torch
import torch.nn as nn 
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch import optim
import torch.nn.functional as F 
from utils.masked_cross_entroy import * 
from utils.config import *
import random 
import numpy as np 
import datetime 
from utils.measures import wer, moses_multi_bleu
import nltk 
import os 
from sklearn.metrics import f1_score

class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use lis of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """
    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix+str(i))

class EncoderMemNN(nn.Module):
    def __init__(self, vocab, embedding_dim, hop, dropout, unk_mask):
        super(EncoderMemNN, self).__init__()
        self.num_vocab = vocab
        self.max_hops = hop
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.unk_mask = unk_mask

        for hop in range(self.max_hops+1):
            C = nn.Embedding(self.num_vocab, embedding_dim, padding_idx=PAD_token)
            C.weight.data.normal_(0, 0.1)
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.softmax = nn.softmax(dim=1)
    def get_state(self, bsz):
        """Get cell states and hidden states"""
        if USE_CUDA:
            return Variable(torch.zeros(bsz, self.embedding_dim)).cuda()
        else:
            return Variable(torch.zeros(bsz, self.embedding_dim))
    def forward(self, story):
        story = story.transpose(0, 1)
        story_size = story.size() # b * m * 3
        if self.unk_mask:
            if(self.training):
                ones = np.ones(story_size[0],story_size[1],story_size[2])
                rand_mask = np.random.binomial([np.ones((story_size[0],story_size[1],story_size[2]))], 1-self.dropout)[0]
                ones[:,:,0] = ones[:,:,0]*rand_mask
                a = Variable(torch.Tensor(ones))
                if USE_CUDA: a = a.cuda()
                story = story*a.long()
        u = [self.get_state(story.size(0))]
        for hop in range(self.max_hops):
            embed_A = self.C[hop](story.contiguous().view(story.szie(0),-1).long()) # b * (m * s) * e
            embed_A = embed_A.view(story_size + (embed_A.size(-1),)) # b * m * s * e
            embed_A = torch.sum(embed_A, 2).squeeze(2)
            m_A = embed_A # b * m * e

            u_temp = u[-1].unsqueeze(1).expand_as(m_A)
            prob = self.softmax(torch.sum(m_A*u_temp, 2)) # b * m
            embed_C = self.C[hop+1](story.contiguous().view(story.size(0), -1).long())
            embed_C = embed_C.view(story_size+(embed_C.size(-1),)) 
            m_C = torch.sum(embed_C, 2).squeeze(2)

            prob = prob.unsqueeze(2).expand_as(m_C)
            o_k = torch.sum(m_C*prob, 1)
            u_k = u[-1] + o_k
            u.append(u_k)
        return u_k



class Mem2Seq(nn.Module):
    def __init__(self, hidden_size, max_len, max_r, lang, path, task, lr, n_layers, dropout, unk_mask):
        super(Mem2Seq, self).__init__()
        self.name = "Mem2Seq"
        self.task = task 
        self.input_size = lang.n_words
        self.output_size = lang.n_words
        self.hidden_size = hidden_size
        self.max_len = max_len
        self.max_r = max_r
        self.lang = lang
        self.lr = lr
        self.n_layers = n_layers
        self.dropout = dropout
        self.unk_mask = unk_mask

        if path:
            if USE_CUDA:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th')
                self.decoder = torch.load(str(path)+'/dec.th')
            else:
                logging.info("MODEL {} LOADED".format(str(path)))
                self.encoder = torch.load(str(path)+'/enc.th',lambda storage, loc: storage)
                self.decoder = torch.load(str(path)+'/dec.th',lambda storage, loc: storage)
        else:
            self.encoder = EncoderMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)
            self.decoder = DecoderrMemNN(lang.n_words, hidden_size, n_layers, self.dropout, self.unk_mask)

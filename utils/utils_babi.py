import json
import torch
import torch.utils.data as data 
import unicodedata
import string 
import re 
import random 
import time 
import math 
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils.config import *
import logging 
import datetime
import ast
from utils.until_temp import entityList

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

MEM_TOKEN_SIZE = 3

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {UNK_token: "UNK", PAD_token: "PAD", EOS_token: "EOS",  SOS_token: "SOS"}
        self.n_words = 4

    def index_words(self, story, trg=False):
        if trg:
            for word in story.split(' '):
                self.index_word(word)
        else:
            for word_triple in story:
                for word in word_triple:
                    self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

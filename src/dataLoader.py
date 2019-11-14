# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch.utils import data
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer

def joinParse(data_file):
    data = data_file.read()
    entireList = []
    partialList = []
    sentences = parse(data)
    for sentence in sentences:
        for word in sentence:
            partialList.append((word['form'], word['upostag'], sentence.metadata))
        entireList.append(partialList)
        partialList = []
    return entireList


class SentimentDataset(data.Dataset):
    '''                                                                                             
    Appends [CLS] and [SEP] token in the beginning and in the end                                   
    to conform to BERT convention, in addition to <pad>                                             
    '''

    def __init__(self, file_path, model_name):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        sents, tags_li, original_sentences = [], [], []
        for sent in tagged:
            words = [word_pos[0] for word_pos in sent]
            tags = [word_pos[1] for word_pos in sent]
            sentence = [word_pos[2] for word_pos in sent]
            sents.append(['[CLS]'] + words + ['[SEP]'])
            tags_li.append(['<pad>'] + tags + ['<pad>'])
            original_sentences.append(sentence)
        self.sents, self.tags_li, self.original_sentences = sents, tags_li, original_sentences

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, idx):
        words, tags, sentences = self.sents[idx], self.tags_li[idx], self.original_sentences[idx]
        x, y = [], []
        is_heads = []
        for word, tag in zip(words, tags):
            tokens = self.tokenizer.tokenize(word) if word not in ('[CLS]', '[SEP]') else [word]
            tokenToId = self.tokenizer.convert_tokens_to_ids(tokens)

            is_head = [1] + [-1] * (len(tokens) - 1)
            tag = [tag] + ['<pad>'] * (len(tokens) - 1)
            tagEach = [tag2idx[each] for each in tag]

            x.extend(tokenToId)
            is_heads.extend(is_head)
            y.extend(tagEach)

        assert len(x) == len(y) == len(is_heads), 'len(x) = {}, len(y) = {}, len(is_heads) = {}'.format(len(x), len(y),
                                                                                                        len(is_heads))
        seqlen = len(y)
        words = ' '.join(words)
        tags = ' '.join(tags)
        return words, x, is_heads, tags, y, seqlen, sentences


def pad(batch):
    f = lambda x: [sample[x] for sample in batch]
    words = f(0)
    is_heads = f(2)
    tags = f(3)
    seqlens = f(-2)
    sentences = f(-1)
    maxlen = np.array(seqlens).max()

    f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch]
    x = f(1, maxlen)
    f = lambda x, seqlen: [sample[x] + [-1] * (seqlen - len(sample[x])) for sample in batch]
    y = f(-3, maxlen)
    f = torch.LongTensor
    return words, f(x), is_heads, tags, f(y), seqlens, sentences


import os
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            token_id = len(self.idx2word) - 1
            self.word2idx[word] = token_id
        else:
            token_id = self.word2idx[word]

        self.counter[token_id] += 1
        self.total += 1
        return token_id

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self._eos = ['<eos>']

    def tokenize(self, path):
        assert os.path.exists(path)

        tokens = 0
        with open(path, 'r') as f:  # Add words to the dictionary
            for line in f:
                words = line.split() + self._eos
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        with open(path, 'r') as f:  # Tokenize file content
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + self._eos
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

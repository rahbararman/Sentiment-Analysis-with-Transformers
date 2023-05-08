'''
Data Loading logic adapted from https://github.com/aladdinpersson/Machine-Learning-Collection with the following licence:

MIT License

Copyright (c) 2020 Aladdin Persson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''

import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset

spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class YelpDataset(Dataset):
    def __init__(self, reviews_file, freq_threshold=5, vocab=None):
        self.df = pd.read_csv(reviews_file, header=None)
        self.df.columns = ['label', 'review']

        self.labels = self.df["label"]
        self.reviews = self.df["review"]

        # Initialize vocabulary and build vocab
        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.reviews.tolist())
        else:
            self.vocab = vocab
    
    def __len__(self):
        return len(self.df)
    
    def get_vocab(self):
        return self.vocab

    
    def __getitem__(self, index):
        review = self.reviews[index]
        label = self.labels[index]-1

        numericalized_review = [self.vocab.stoi["<SOS>"]]
        numericalized_review += self.vocab.numericalize(review)
        numericalized_review.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(numericalized_review), label

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        reviews = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        reviews = pad_sequence(reviews, batch_first=True, padding_value=self.pad_idx)

        return reviews, torch.tensor(labels)


def get_loader(
    reviews_file,
    batch_size=32,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
    vocab=None
):
    dataset = YelpDataset(reviews_file, vocab=vocab)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset

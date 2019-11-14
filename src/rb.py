import torch
import argparse
from torch.utils import data
import numpy as np
import time
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.rnn as rnn
from torchnlp.datasets import imdb_dataset
from pytorch_pretrained_bert import BertTokenizer, BertModel
import sys

"""
ToDo:
1. sort the dataset by text length
2. use the max length of the batch in sort batch instead of the maximum length of all text
"""


class SentDataset(data.Dataset):
    '''
    Appends [CLS] and [SEP] token in the beginning and in the end
    to conform to BERT convention
    '''

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]['text'], self.data[idx]['sentiment']
        assert label in {'pos', 'neg'}
        y = 1 if label == 'pos' else 0
        tokens = tokenizer.tokenize(text)[:MAX_LEN - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        raw_tokenId = tokenizer.convert_tokens_to_ids(tokens)
        length = len(raw_tokenId)
        raw_tokenId += [0] * (MAX_LEN - length)
        mask = [1] * length + [0] * (MAX_LEN - length)
        assert len(mask) == len(raw_tokenId)
        return raw_tokenId, y, length, text, mask


def sort_batch(batch):
    lengths = [example[2] for example in batch]
    indexes = np.argsort(lengths).tolist()[::-1]
    f = lambda x: [batch[idx][x] for idx in indexes]
    tokenid, y, length, text, mask = f(0), f(1), f(2), f(3), f(4)
    tokenid = torch.tensor(tokenid, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.float)
    mask = torch.tensor(mask, dtype=torch.long)
    return tokenid, y, length, text, mask


class BERT_biLSTM(nn.Module):
    def __init__(self, hidden_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-{}-cased'.format('large' if LARGE else 'base'))
        for param in self.bert.parameters():
            param.requires_grad = False

        self.rnn = nn.LSTM(1024 if LARGE else 768,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout)

        self.fc = nn.Linear(hidden_dim * 2, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, tokenids, text_lengths, attention_masks):
        tokenids = tokenids.to(device)
        attention_masks = attention_masks.to(device)
        self.bert.eval()

        # start_time = time.time()

        with torch.no_grad():
            # enc, _ = self.bert(tokenids, attention_mask=attention_masks, output_all_encoded_layers = False)
            encoded_layers, _ = self.bert(tokenids, attention_mask=attention_masks)
            enc = encoded_layers[-1]

        # print("bert time:{}".format(time.time() - start_time))
        # start_time = time.time()

        packed_enc = rnn.pack_padded_sequence(enc, text_lengths, batch_first=True)
        packed_output, (hidden, cell) = self.rnn(packed_enc)

        # print("rnn time:{}".format(time.time() - start_time))
        # start_time = time.time()

        output, output_lengths = rnn.pad_packed_sequence(packed_output, batch_first=True)

        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))

        predict = self.fc(hidden)

        # print("rest time:{}".format(time.time() - start_time))

        return predict


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion, epoch):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    batch_acc = 0
    start_time = time.time()

    for i, batch in enumerate(iterator):
        optimizer.zero_grad()

        tokenIds, labels, text_lengths, texts, masks = batch

        predictions = model(tokenIds, text_lengths, masks).squeeze(1)

        labels = labels.to(device)
        loss = criterion(predictions, labels)

        acc = binary_accuracy(predictions, labels)

        if i % 10 == 0:
            print("epoch:{}, step: {}, acc: {}, time: {}s".format(epoch, i, round(batch_acc / 10, 3),
                                                                  round(time.time() - start_time, 3)))
            start_time = time.time()
            batch_acc = 0

        loss.backward()

        optimizer.step()

        batch_acc += acc.item()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


parser = argparse.ArgumentParser()
parser.add_argument("--large", action='store_true')
args = parser.parse_args()

LARGE = args.large
MAX_LEN = 256  # time: 128(10s), 256(20s), 500(40s)
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
N_EPOCHS = 1

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

tokenizer = BertTokenizer.from_pretrained('bert-{}-cased'.format('large' if LARGE else 'base'), do_lower_case=False)


def main():
    train_data, test_data = imdb_dataset(train=True, test=True)
    train_data = random.sample(train_data, 5000)

    train_dataset, test_dataset = SentDataset(train_data), SentDataset(test_data)
    trainIteration = data.DataLoader(dataset=train_dataset, collate_fn=sort_batch, batch_size=50, shuffle=True)

    model = BERT_biLSTM(HIDDEN_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT)
    model = model.to(device)
    if device == 'cuda': model = nn.DataParallel(model)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    best_valid_loss = float('inf')
    print("starting training")

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, trainIteration, optimizer, criterion, epoch)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')


if __name__ == "__main__":
    main()
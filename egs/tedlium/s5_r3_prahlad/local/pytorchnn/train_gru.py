import torch
from torch import nn
from livelossplot import PlotLosses
import numpy as np
from collections import defaultdict

def read_vocab(path):
    r"""Read vocabulary.

    Args:
        path (str): A file with a word and its integer index per line.

    Returns:
        A vocabulary represented by a map from string to int (starting from 0).
    """

    word2idx = {}
    idx2word = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.split()
            assert len(word) == 2
            word = word[0]
            if word not in word2idx:
                idx2word.append(word)
                word2idx[word] = len(idx2word) - 1
    return word2idx

def load_sents(path):
    r"""Read word sentences that represent hypotheses of utterances.

    Assume the input file format is "utterance-id word-sequence" in each line:
        en_4156-A_030185-030248-1 oh yeah
        en_4156-A_030470-030672-1 well i'm going to have mine and two more classes
        en_4156-A_030470-030672-2 well i'm gonna have mine and two more classes
        ...

    Args:
        path (str): A file of word sentences in the above format.

    Returns:
        The sentences represented by a map from a string (utterance-id) to
        a list of strings (hypotheses).
    """

    sents = defaultdict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            try:
                key, hyp = line.split(' ', 1)
            except ValueError:
                key = line
                hyp = ' '
            key = key.rsplit('-', 1)[0]
            if key not in sents:
                sents[key] = [hyp]
            else:
                sents[key].append(hyp)
    return sents

def get_input_and_target(hyps, vocab):
    r"""Convert hypotheses to lists of integers, with input and target separately.

    Args:
        hyps (str): Hypotheses, with words separated by spaces, e.g.'hello there'
        vocab:      A map from string to int, e.g. {'<s>':0, 'hello':1,
                    'there':2, 'apple':3, ...}

    Returns:
        A pair of lists, one with the integerized input sequence, one with the
        integerized output (target) sequence: in this case ([0 1 2], [1 2 0]),
        since the input sequence has '<s>' at the beginning and the output
        sequence has '<s>' at the end. Words that are not in the vocabulary are
        mapped to a special oov symbol, which is expected to be in the vocabulary.
    """
    batch_size = len(hyps)
    assert batch_size > 0

    # Preprocess input and target sequences
    inputs, outputs = [], []
    for hyp in hyps:
        input_string = '<s>' + ' ' + hyp
        output_string = hyp + ' ' + '<s>'
        input_ids, output_ids = [], []
        for word in input_string.split():
            try:
                input_ids.append(vocab[word])
            except KeyError:
                input_ids.append(vocab['<unk>'])
        for word in output_string.split():
            try:
                output_ids.append(vocab[word])
            except KeyError:
                output_ids.append(vocab['<unk>'])
        inputs.append(input_ids)
        outputs.append(output_ids)

    batch_lens = [len(seq) for seq in inputs]
    seq_lens = torch.LongTensor(batch_lens)
    max_len = max(batch_lens)

    # Zero padding for input and target sequences.
    data = torch.LongTensor(batch_size, max_len).zero_()
    target = torch.LongTensor(batch_size, max_len).zero_()
    for idx, seq_len in enumerate(batch_lens):
        data[idx, :seq_len] = torch.LongTensor(inputs[idx])
        target[idx, :seq_len] = torch.LongTensor(outputs[idx])
    data = data.t().contiguous()
    target = target.t().contiguous().view(-1)
    return data, target, seq_lens, max_len

def get_batches(type, train_sents, vocab):
  main_data = defaultdict()
  for idx, key in enumerate(train_sents.keys()):
    data = defaultdict()
    batch_size = len(train_sents[key])
    # Dimension of input data is [seq_len, batch_size]
    input_data, targets, seq_lens, max_len = get_input_and_target(train_sents[key], vocab)
    data['sent'] = input_data
    data['sent_length'] = max_len
    main_data[idx] = data
  return main_data

embedding_size = 850
hidden_size = 850
vocab_size = 152218

class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.GRU(embedding_size, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.crossentropy = nn.CrossEntropyLoss()

    def forward(self, data):
        # data is the dict returned by ``dataloader.get_batch``
        sent = data['sent']
        sent_length = data['sent_length']
        # sent is a LongTensor whose shape is (batch_size, max(sent_length))
        # sent_length is a list whose size is (batch_size)

        incoming = self.embedding_layer(sent)
        # incoming: (batch_size, max(sent_length), embedding_size)
        incoming, _ = self.rnn(incoming)
        # incoming: (batch_size, max(sent_length), hidden_size)
        incoming = self.output_layer(incoming)
        # incoming: (batch_size, max(sent_length), dataloader.frequent_vocab_size)

        loss = []
        for i, length in enumerate(sent_length):
            if length > 1:
                loss.append(self.crossentropy(incoming[i, :length-1], sent[i, 1:length]))
                # every time step predict next token

        data["gen_log_prob"] = nn.LogSoftmax(dim=-1)(incoming)

        if len(loss) > 0:
            return torch.stack(loss).mean()
        else:
            return 0

def main():
    vocab = read_vocab('/content/drive/MyDrive/data/pytorchnn/words.txt')
    vocab_size = len(vocab.keys())
    print(vocab_size)
    train_sents = load_sents('/content/drive/MyDrive/data/pytorchnn/train.txt')

    net = LanguageModel()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    epoch_num = 100
    plot = PlotLosses()

    for j in range(epoch_num):
        loss_arr = []
        for i, data in enumerate(get_batches("train", train_sents, vocab)):
            # convert numpy to torch.LongTensor
            # data['sent'] = torch.LongTensor(data['sent'])
            net.zero_grad()
            loss = net(data, vocab_size)
            loss_arr.append(loss.tolist())
            loss.backward()
            optimizer.step()
            #if i >= 40:
                #break # break for shorten time of an epoch
        plot.update({"loss": np.mean(loss_arr)})
        plot.draw()
        print("epoch %d/%d" % (j+1, epoch_num))

if __name__ == '__main__':
    main()
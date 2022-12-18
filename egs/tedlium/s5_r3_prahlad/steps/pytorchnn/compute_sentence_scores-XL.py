# Copyright 2020    Ke Li

""" This script computes sentence scores in a batch computation mode with a
    PyTorch-trained neural LM.
    It is called by steps/pytorchnn/lmrescore_{nbest, lattice}_pytorchnn.sh
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def get_input_and_target(args, hyps, tokenizer, vocab):
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
    inputs = []
    for hyp in hyps:
        input_ids = tokenizer(args.sent_boundary + ' ' + hyp, return_tensors="pt").to(device)
        # input_ids, output_ids = [], []
        # for word in input_string.split():
        #     try:
        #         input_ids.append(vocab[word])
        #     except KeyError:
        #         input_ids.append(vocab[args.oov])
        # for word in output_string.split():
        #     try:
        #         output_ids.append(vocab[word])
        #     except KeyError:
        #         output_ids.append(vocab[args.oov])
        inputs.append(input_ids)

    # batch_lens = [len(seq) for seq in inputs]
    #seq_lens = torch.LongTensor(batch_lens)
    # max_len = max(batch_lens)

    # # Zero padding for input and target sequences.
    # data = torch.LongTensor(batch_size, max_len).zero_()
    # target = torch.LongTensor(batch_size, max_len).zero_()
    # for idx, seq_len in enumerate(batch_lens):
    #     data[idx, :seq_len] = torch.LongTensor(inputs[idx])
    #     target[idx, :seq_len] = torch.LongTensor(outputs[idx])
    # data = data.t().contiguous()
    # target = target.t().contiguous().view(-1)
    return inputs


def compute_sentence_score(model, criterion, ntokens, inputs,
                           model_type='LSTM', hidden=None):
    r"""Compute neural language model scores of hypotheses of an utterance.

    Args:
        model:      A neural language model.
        criterion:  Training criterion of a neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        data:       Integerized input sentences (hypotheses).
        target:     Integerized target sentences for data.
        model_type: Model type, e.g. LSTM or Transformer or others.
        hidden:     Initial hidden state for a recurrent-typed model (optional).

    Returns:
        The scores (negative log-likelihood) of words in input hypotheses.
        If the model is recurrent-typed, the function has an extra output:
        the last hidden state from the best hypothesis for an utterance.
    """

    with torch.no_grad():
        losses = []
        mems = None
        for input_tokenizer in inputs:
            print(input_tokenizer)
            output = model(**input_tokenizer, labels=input_tokenizer['input_ids'], mems=mems)
            mems = output.mems
            losses.append(output.losses.cpu().detach().flatten().numpy())

        loss_lens = [len(loss) for loss in losses]
        max_losslen = max(loss_lens)

        for i, loss in enumerate(losses):
            losses[i] = np.pad(loss, (0, max_losslen-len(loss)), 'constant')

        sent_scores = np.stack(losses, axis = 0)
    return sent_scores, loss_lens

def compute_scores(args, sents, model, criterion, tokenizer, ntokens, vocab, model_type='LSTM'):
    r"""Compute neural language model scores of hypotheses for all utterances.

    Args:
        sents:      Hypotheses for all utterances represented by a map from
                    a string (utterance-id) to a list of strings.
        model:      A neural language model.
        criterion:  Training criterion of the neural language model, e.g.
                    cross entropy.
        ntokens:    Vocabulary size.
        model_type: Model type, e.g. LSTM or Transformer or others.

    Returns:
        The hypotheses and corresponding neural language model scores for all
        utterances.
    """

    # Turn on evaluation mode which disables dropout.
    model.eval()
    sents_and_scores = defaultdict()
    for idx, key in enumerate(sents.keys()):
        batch_size = len(sents[key])
        # Dimension of input data is [seq_len, batch_size]
        inputs = get_input_and_target(args, sents[key], tokenizer, vocab)
        
        scores, seq_lens = compute_sentence_score(model, criterion, ntokens, inputs,
                                            model_type)
        for idx, hyp in enumerate(sents[key]):
            if key in sents_and_scores:
                sents_and_scores[key].append((hyp, scores[idx][:seq_lens[idx]]))
            else:
                sents_and_scores[key] = [(hyp, scores[idx][:seq_lens[idx]])]

    return sents_and_scores


def write_scores(sents_and_scores, path):
    r"""Write out neural language model scores for all hypotheses in the
        following format:
        en_4156-A_030185-030248-1 2.7702 1.9545 0.9442
        en_4156-A_030470-030672-1 3.6918 3.7159 4.1794 0.1375 2.3944 9.3834 4.5469 7.0772 3.6172 7.2183 2.1540
        en_4156-A_030470-030672-2 3.6918 3.7159 4.5248 2.3689 8.9368 4.2876 7.0702 3.0812 7.5044 2.2388
        ...

    Args:
        sents_and_scores: The hypotheses and scores represented by a map from
                          a string to a pair of a hypothesis and scores.
        path (str):       A output file of scores in the above format.
    """

    with open(path, 'w', encoding='utf-8') as f:
        for key in sents_and_scores.keys():
            for idx, (_, score_list) in enumerate(sents_and_scores[key], 1):
                current_key = '-'.join([key, str(idx)])
                f.write('{} '.format(current_key))
                for score in score_list:
                    f.write("{0:.4f} ".format(score))
                f.write('\n')
    print("Write neural LM scores to %s" % path)


def main():
    parser = argparse.ArgumentParser(description="Compute word scores of"
                                     "hypotheses for each utterance in parallel"
                                     "with a PyTorch-trained neural language model.")
    parser.add_argument('--infile', type=str, required=True,
                        help="Word hypotheses generated from a lattice.")
    parser.add_argument('--outfile', type=str, required=True,
                        help="Output file with neural language model scores"
                        "for input word hypotheses.")
    parser.add_argument('--vocabulary', type=str, required=True,
                        help="Vocabulary used for neural language model training.")
    parser.add_argument('--model-path', type=str, required=True,
                        help="Path to a pretrained neural language model.")
    parser.add_argument('--model', type=str, default='LSTM',
                        help='Network type. Can be RNN, LSTM or Transformer.')
    parser.add_argument('--emsize', type=int, default=200,
                        help='Size of word embeddings.')
    parser.add_argument('--nhid', type=int, default=200,
                        help='Number of hidden units per layer.')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--nhead', type=int, default=2,
                        help='Number of heads in a Transformer model.')
    parser.add_argument('--oov', type=str, default='<unk>',
                        help='Out of vocabulary word.')
    parser.add_argument('--sent-boundary', type=str, default='<s>',
                        help='Sentence boundary symbol.')
    args = parser.parse_args()
    assert os.path.exists(args.infile), "Path for input word sequences does not exist."
    assert os.path.exists(args.vocabulary), "Vocabulary path does not exist."
    assert os.path.exists(args.model_path), "Model path does not exist."

    print("Load vocabulary.")
    vocab = read_vocab(args.vocabulary)
    ntokens = len(vocab)
    
    print("Load Transformer XL model.")
    model = TransfoXLLMHeadModel.from_pretrained(args.model_path).to(device)
    tokenizer = TransfoXLTokenizer.from_pretrained("transfo-xl-wt103")

    criterion = nn.CrossEntropyLoss(reduction='none')
    print("Load input word hypotheses.")
    sents = load_sents(args.infile)
    print("Compute word scores with a ", args.model, " model.")
    sents_and_scores = compute_scores(args, sents, model, criterion, tokenizer, ntokens, vocab,
                                      model_type=args.model)
    print("Write out word scores.")
    write_scores(sents_and_scores, args.outfile)

if __name__ == '__main__':
    main()

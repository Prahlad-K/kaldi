#!/usr/bin/env bash

# Copyright 2016  Vimal Manohar
#           2016  Johns Hopkins University (author: Daniel Povey)
# Apache 2.0

# This script demonstrates how to re-segment training data selecting only the
# "good" audio that matches the transcripts.
# The basic idea is to decode with an existing in-domain acoustic model, and a
# biased language model built from the reference, and then work out the
# segmentation from a ctm like file.

# For nnet3 and chain results after cleanup, see the scripts in
# local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh

# GMM Results for speaker-independent (SI) and speaker adaptive training (SAT) systems on dev and test sets
# [will add these later].

set -e
set -o pipefail
set -u

stage=0
cleanup_stage=0
data=data/train
cleanup_affix=cleaned
srcdir=exp/tri3
nj=100
decode_nj=16
decode_num_threads=4

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh

cleaned_data=${data}_${cleanup_affix}

dir=${srcdir}_${cleanup_affix}_work
cleaned_dir=${srcdir}_${cleanup_affix}

if [ $stage -le 1 ]; then
  # This does the actual data cleanup.
  steps/cleanup/clean_and_segment_data.sh --stage $cleanup_stage --nj $nj --cmd "$train_cmd" \
    $data data/lang $srcdir $dir $cleaned_data
fi

if [ $stage -le 2 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    $cleaned_data data/lang $srcdir ${srcdir}_ali_${cleanup_affix}
fi

if [ $stage -le 3 ]; then
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 $cleaned_data data/lang ${srcdir}_ali_${cleanup_affix} ${cleaned_dir}
fi

if [ $stage -le 4 ]; then
  # Test with the models trained on cleaned-up data.
  utils/mkgraph.sh data/lang ${cleaned_dir} ${cleaned_dir}/graph

  # for dset in dev test; do
  #   steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd" --num-threads $decode_num_threads \
  #      ${cleaned_dir}/graph data/${dset} ${cleaned_dir}/decode_${dset}
  #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
  #      data/${dset} ${cleaned_dir}/decode_${dset} ${cleaned_dir}/decode_${dset}_rescore
  # done
fi

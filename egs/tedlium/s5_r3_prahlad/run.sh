#!/usr/bin/env bash
#
# Based mostly on the Switchboard recipe. The training database is TED-LIUM,
# it consists of TED talks with cleaned automatic transcripts:
#
# https://lium.univ-lemans.fr/ted-lium3/
# http://www.openslr.org/resources (Mirror).
#
# The data is distributed under 'Creative Commons BY-NC-ND 3.0' license,
# which allow free non-commercial use, while only a citation is required.
#
# Copyright  2014  Nickolay V. Shmyrev
#            2014  Brno University of Technology (Author: Karel Vesely)
#            2016  Vincent Nguyen
#            2016  Johns Hopkins University (Author: Daniel Povey)
#            2018  François Hernandez
#
# Apache 2.0
#

. ./cmd.sh
. ./path.sh


set -e -o pipefail -u

nj=8
decode_nj=8   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.
stage=18
train_rnnlm=false
train_lm=false

. utils/parse_options.sh # accept options


echo "Started running run.sh"

# Data preparation
if [ $stage -le 0 ]; then
  echo "Stage 0 start"
  local/download_data.sh
  echo "Stage 0 end"
fi

if [ $stage -le 1 ]; then
  echo "Stage 1 start"
  echo "Stage 1: Preparing data start"
  local/prepare_data.sh
  echo "Stage 1: Preparing data end"
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  echo "Stage 1: Modifying speaker info start"
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  echo "Stage 1: Modifying speaker info end"
  echo "Stage 1 end"
  done
fi


if [ $stage -le 2 ]; then
  echo "Stage 2 start"
  local/prepare_dict.sh
  echo "Stage 2 end"
fi

if [ $stage -le 3 ]; then
  echo "Stage 3 start"
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
  echo "Stage 3 end"
fi

if [ $stage -le 4 ]; then
  echo "Stage 4 start"
  # later on we'll change this script so you have the option to
  # download the pre-built LMs from openslr.org instead of building them
  # locally.
  if $train_lm; then
    local/ted_train_lm.sh
  else
    local/ted_download_lm.sh
  fi
  echo "Stage 4 end"
fi

if [ $stage -le 5 ]; then
  echo "Stage 5 start"
  local/format_lms.sh
  echo "Stage 5 end"
fi

# Feature extraction
if [ $stage -le 6 ]; then
  echo "Stage 6 start"
  for set in test dev train; do
    dir=data/$set
    steps/make_mfcc_pitch.sh --nj 30 --cmd "$train_cmd" $dir
    steps/compute_cmvn_stats.sh $dir
  done
  echo "Stage 6 end"
fi

# Now we have 452 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 7 ]; then
  echo "Stage 7 start"
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  echo "Stage 7 finished subset"
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
  echo "Stage 7 end"
fi

# Train
if [ $stage -le 8 ]; then
  echo "Stage 8 start"
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono
  echo "Stage 8 end"
fi

if [ $stage -le 9 ]; then
  echo "Stage 9 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali
  echo "Stage 9 aligning done"
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang_nosp exp/mono_ali exp/tri1
  echo "Stage 9 end"
fi

if [ $stage -le 10 ]; then
  echo "Stage 10 start"
  utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph_nosp
  echo "Stage 10 mkgraph done"
  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
  #for dset in dev test; do
  #  steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #    exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset}
  #  echo "Stage 10 decoding done"
  #  steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
  #     data/${dset} exp/tri1/decode_nosp_${dset} exp/tri1/decode_nosp_${dset}_rescore
  #  echo "Stage 10 rescoring done"
  #done
  echo "Stage 10 end"
fi

if [ $stage -le 11 ]; then
  echo "Stage 11 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali
  echo "Stage 11 aligning done"
  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1_ali exp/tri2
  echo "Stage 11 train lda mllt done"
  echo "Stage 11 end"
fi

if [ $stage -le 12 ]; then
  echo "Stage 12 start"
  utils/mkgraph.sh data/lang_nosp exp/tri2 exp/tri2/graph_nosp
  echo "Stage 12 making graph done"
  # for dset in dev test; do
  #   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset}
  #   echo "Stage 12 decoding done"
  #   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
  #      data/${dset} exp/tri2/decode_nosp_${dset} exp/tri2/decode_nosp_${dset}_rescore
  #   echo "Stage 12 rescoring done"
  # done
  echo "Stage 12 end"
fi

if [ $stage -le 13 ]; then
  echo "Stage 13 start"
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2
  echo "Stage 13 get prons done"
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict
  echo "Stage 13 dict dir adding pronouns done"
  echo "Stage 13 end"
fi

if [ $stage -le 14 ]; then
  echo "Stage 14 start"
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/
  echo "Stage 14 preparing lang done"

  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph
  echo "Stage 14 making graph done"

  # for dset in dev test; do
  #   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri2/graph data/${dset} exp/tri2/decode_${dset}
  #   echo "Stage 14 decoding done"
  #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
  #      data/${dset} exp/tri2/decode_${dset} exp/tri2/decode_${dset}_rescore
  #   echo "Stage 14 rescoring done"
  # done
  echo "Stage 14 end"
fi

if [ $stage -le 15 ]; then
  echo "Stage 15 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali
  echo "Stage 15 aligning done"
  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3
  echo "Stage 15 sat done"
  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph
  echo "Stage 15 making graph done"
  # for dset in dev test; do
  #   steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri3/graph data/${dset} exp/tri3/decode_${dset}
  #   echo "Stage 15 decoding done"
  #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
  #      data/${dset} exp/tri3/decode_${dset} exp/tri3/decode_${dset}_rescore
  #   echo "Stage 15 rescoring done"
  # done
  echo "Stage 15 end"
fi

if [ $stage -le 16 ]; then
  echo "Stage 16 start"
  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh
  echo "Stage 16 end"
fi

# echo "Need to enable GPU before moving forward with program (end of CPU portion)."
# exit 1


if [ $stage -le 17 ]; then
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  echo "start run stage 17"
  local/chain/run_tdnn.sh
  echo "run stage 17 DONE"
fi

if [ $stage -le 18 ]; then
  # You can either train your own rnnlm or download a pre-trained one
  if $train_rnnlm; then
    local/rnnlm/tuning/run_lstm_tdnn_a.sh
    local/rnnlm/average_rnnlm.sh
  else
    local/ted_download_rnnlm.sh
  fi
fi

if [ $stage -le 19 ]; then
  # Here we rescore the lattices generated at stage 17
  rnnlm_dir=exp/rnnlm_lstm_tdnn_a_averaged
  lang_dir=data/lang_chain
  ngram_order=4

  for dset in dev test; do
    data_dir=data/${dset}_hires
    # decoding_dir=exp/chain_cleaned/tdnnf_1a/decode_${dset}
    decoding_dir=exp/chain_cleaned_1d/tdnn1d_sp/decode_${dset}
    suffix=$(basename $rnnlm_dir)
    output_dir=${decoding_dir}_$suffix

    rnnlm/lmrescore_pruned.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight 0.5 --max-ngram-order $ngram_order \
      $lang_dir $rnnlm_dir \
      $data_dir $decoding_dir \
      $output_dir
  done
fi


echo "$0: success."
exit 0

#!/usr/bin/env bash
# *********** AUTHOR: Prahlad Koratamaddi, UNI: pk2743. 18th December, 2022 ****************
# Extending the work done by the following team of TED-LIUM


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

nj=35
decode_nj=38   # note: should not be >38 which is the number of speakers in the dev set
               # after applying --seconds-per-spk-max 180.  We decode with 4 threads, so
               # this will be too many jobs if you're using run.pl.

stage=19                  # proceeds to decoding at the last stage

# pk2743: including the following additional options
model="pytorch_transformer" # can choose between pytorch_transformer [default] or transformer_xl or gcnnlm
nbest=true                # if true [default], does nbest list rescoring otherwise, does pruning + lattice rescoring 
train_nnlm=true          # if false [default] proceeds to decoding without training the NNLM
decode_on_tedlium=true    # if true [default], decodes on the tedlium test dataset, otherwise on the LibriSpeech test-other dataset         

train_lm=false

. utils/parse_options.sh # accept options

# Data preparation
if [ $stage -le 0 ]; then
  echo "Stage 0 start"
  local/download_data.sh
fi

if [ $stage -le 1 ]; then
  echo "Stage 1 start"
  local/prepare_data.sh
  # Split speakers up into 3-minute chunks.  This doesn't hurt adaptation, and
  # lets us use more jobs for decoding etc.
  # [we chose 3 minutes because that gives us 38 speakers for the dev data, which is
  #  more than our normal 30 jobs.]
  for dset in dev test train; do
    utils/data/modify_speaker_info.sh --seconds-per-spk-max 180 data/${dset}.orig data/${dset}
  done
fi


if [ $stage -le 2 ]; then
  echo "Stage 2 start"
  local/prepare_dict.sh
fi

if [ $stage -le 3 ]; then
  echo "Stage 3 start"
  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_nosp data/lang_nosp
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
fi

if [ $stage -le 5 ]; then
  echo "Stage 5 start"
  local/format_lms.sh
fi

# Feature extraction
if [ $stage -le 6 ]; then
  echo "Stage 6 start"
  for set in test dev train; do
    dir=data/$set
    steps/make_mfcc.sh --nj 30 --cmd "$train_cmd" $dir
    steps/compute_cmvn_stats.sh $dir
  done
fi

# Now we have 452 hours of training data.
# Well create a subset with 10k short segments to make flat-start training easier:
if [ $stage -le 7 ]; then
  echo "Stage 7 start"
  utils/subset_data_dir.sh --shortest data/train 10000 data/train_10kshort
  utils/data/remove_dup_utts.sh 10 data/train_10kshort data/train_10kshort_nodup
fi

# Train
if [ $stage -le 8 ]; then
  echo "Stage 8 start"
  steps/train_mono.sh --nj 20 --cmd "$train_cmd" \
    data/train_10kshort_nodup data/lang_nosp exp/mono
fi

if [ $stage -le 9 ]; then
  echo "Stage 9 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/mono exp/mono_ali
  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 30000 data/train data/lang_nosp exp/mono_ali exp/tri1
fi

if [ $stage -le 10 ]; then
  echo "Stage 10 start"
  utils/mkgraph.sh data/lang_nosp exp/tri1 exp/tri1/graph_nosp

  # The slowest part about this decoding is the scoring, which we can't really
  # control as the bottleneck is the NIST tools.
  # for dset in dev test; do
  #   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri1/graph_nosp data/${dset} exp/tri1/decode_nosp_${dset}
  #   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
  #      data/${dset} exp/tri1/decode_nosp_${dset} exp/tri1/decode_nosp_${dset}_rescore
  # done
fi

if [ $stage -le 11 ]; then
  echo "Stage 11 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    4000 50000 data/train data/lang_nosp exp/tri1_ali exp/tri2
fi

if [ $stage -le 12 ]; then
  echo "Stage 12 start"
  utils/mkgraph.sh data/lang_nosp exp/tri2 exp/tri2/graph_nosp
  # for dset in dev test; do
  #   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri2/graph_nosp data/${dset} exp/tri2/decode_nosp_${dset}
  #   steps/lmrescore_const_arpa.sh  --cmd "$decode_cmd" data/lang_nosp data/lang_nosp_rescore \
  #      data/${dset} exp/tri2/decode_nosp_${dset} exp/tri2/decode_nosp_${dset}_rescore
  # done
fi

if [ $stage -le 13 ]; then
  echo "Stage 13 start"
  steps/get_prons.sh --cmd "$train_cmd" data/train data/lang_nosp exp/tri2
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
    data/local/dict_nosp exp/tri2/pron_counts_nowb.txt \
    exp/tri2/sil_counts_nowb.txt \
    exp/tri2/pron_bigram_counts_nowb.txt data/local/dict
fi

if [ $stage -le 14 ]; then
  echo "Stage 14 start"
  utils/prepare_lang.sh data/local/dict "<unk>" data/local/lang data/lang
  cp -rT data/lang data/lang_rescore
  cp data/lang_nosp/G.fst data/lang/
  cp data/lang_nosp_rescore/G.carpa data/lang_rescore/

  utils/mkgraph.sh data/lang exp/tri2 exp/tri2/graph

  # for dset in dev test; do
  #   steps/decode.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri2/graph data/${dset} exp/tri2/decode_${dset}
  #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
  #      data/${dset} exp/tri2/decode_${dset} exp/tri2/decode_${dset}_rescore
  # done
fi

if [ $stage -le 15 ]; then
  echo "Stage 15 start"
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train data/lang exp/tri2_ali exp/tri3

  utils/mkgraph.sh data/lang exp/tri3 exp/tri3/graph

  # for dset in dev test; do
  #   steps/decode_fmllr.sh --nj $decode_nj --cmd "$decode_cmd"  --num-threads 4 \
  #     exp/tri3/graph data/${dset} exp/tri3/decode_${dset}
  #   steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" data/lang data/lang_rescore \
  #      data/${dset} exp/tri3/decode_${dset} exp/tri3/decode_${dset}_rescore
  # done
fi

if [ $stage -le 16 ]; then
  echo "Stage 16 start"
  # this does some data-cleaning.  It actually degrades the GMM-level results
  # slightly, but the cleaned data should be useful when we add the neural net and chain
  # systems.  If not we'll remove this stage.
  local/run_cleanup_segmentation.sh
fi

if [ $stage -le 17 ]; then
  echo "Stage 17 start"
  # This will only work if you have GPUs on your system (and note that it requires
  # you to have the queue set up the right way... see kaldi-asr.org/doc/queue.html)
  local/chain/run_tdnn.sh
fi

if [ $stage -le 18 ]; then
  echo "Stage 18 start"
  # pk2743: Train the NN LM or if already trained proceed
  if $train_nnlm; then
    if [[ "$model" == "pytorch_transformer" ]]; then
      echo "Training the Transformer NNLM............."
      #local/pytorchnn/run_nnlm.sh
    fi
    if [[ "$model" == "transformer_xl" ]]; then
      echo "Training the Transformer-XL NNLM............."
      python3 local/pytorchnn/save_transformer_model.py
    fi
    if [[ "$model" == "gcnnlm" ]]; then
      echo "Training the Gated Convolutional NNLM............."
      python3 local/pytorchnn/save_fairseq_gcnn_model.py
    fi
  fi
fi

if [ $stage -le 19 ]; then
  echo "Stage 19 start"
  # Here we rescore the lattices generated at stage 17

  # pk2743: based on the options declared at the top of the script
  # the decoding process is executed
  tnnlm_dir=exp/$model
  vocab_data_dir=data/pytorchnn
  ngram_order=4

  if [[ "$model" == "pytorch_transformer" ]]; then
    echo "Decoding with the Transformer NNLM............."

    # pk2743: Defining the Transformer NNLM model architecture + other params
    model_type=Transformer
    embedding_dim=768
    hidden_dim=768
    nlayers=8
    nhead=8
    pytorch_path=exp/pytorch_transformer
    nn_model=$pytorch_path/model.pt
    nbest_script=lmrescore_nbest_pytorchnn.sh
    lattice_script=lmrescore_lattice_pytorchnn.sh
    oov='<UNK>' # Symbol for out-of-vocabulary words

  fi
  if [[ "$model" == "transformer_xl" ]]; then
    echo "Decoding with the Transformer-XL NNLM............."

    # pk2743: Defining the Transformer-XL NNLM model architecture + other params
    model_type=Transformer-XL
    embedding_dim=1024
    hidden_dim=4096
    nlayers=18
    nhead=16
    pytorch_path=exp/transformer_xl
    nn_model=$pytorch_path/
    nbest_script=lmrescore_nbest_pytorchnn-XL.sh
    lattice_script=lmrescore_lattice_pytorchnn-XL.sh
    oov='<UNK>' # Symbol for out-of-vocabulary words

  fi
  if [[ "$model" == "gcnnlm" ]]; then
    echo "Decoding with the Gated Convolutional NNLM............."

    # pk2743: Defining the Gated Convolutional NNLM Model Architecture + other params
    model_type=GCNN
    embedding_dim=280
    hidden_dim=850
    nlayers=8
    nhead=16
    pytorch_path=exp/gcnnlm
    nn_model=$pytorch_path/
    nbest_script=lmrescore_nbest_pytorchnn-gcnn.sh
    lattice_script=lmrescore_lattice_pytorchnn-gcnn.sh
    oov='<UNK>' # Symbol for out-of-vocabulary words
  fi
  
  # pk2743: The architecture has been defined, proceeding to prepare params related to dataset
  if $decode_on_tedlium; then
    dset=test
    lang_dir=data/lang_chain
    data_dir=data/${dset}_hires
    decoding_dir=exp/chain_cleaned_1d/tdnn1d_sp/decode_${dset}
    suffix=$(basename $tnnlm_dir)
  else
    # must decode on librispeech!
    dset=dev_clean_2
    lang_dir=data/lang_test_tgsmall
    data_dir=data/${dset}_hires
    decoding_dir=exp/tri3b/decode_tgsmall_$dset
    suffix=$(basename $tnnlm_dir)
  fi

  # pk2743: the dataset params are also defined, proceeding to decode!
  if $nbest; then
    output_dir=${decoding_dir}_${suffix}_nbest
    steps/pytorchnn/$nbest_script \
        --cmd "$decode_cmd --max-jobs-run 1" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.7 \
        --oov-symbol "'$oov'" \
        --stage 7 \
        $lang_dir $nn_model $vocab_data_dir/words.txt \
        $data_dir $decoding_dir \
        $output_dir
  else
    output_dir=${decoding_dir}_${suffix}_lattice
    steps/pytorchnn/$lattice_script \
        --cmd "$decode_cmd --max-jobs-run 1" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight 0.7 \
        --beam 4 \
        --epsilon 0.5 \
        --oov-symbol "'$oov'" \
        --stage 3 \
        $lang_dir $nn_model $vocab_data_dir/words.txt \
        $data_dir $decoding_dir \
        $output_dir
  fi
fi

echo "$0: success."
exit 0

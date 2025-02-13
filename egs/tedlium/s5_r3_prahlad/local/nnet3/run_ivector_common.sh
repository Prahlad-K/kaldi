#!/usr/bin/env bash

set -e -o pipefail


# This script is called from local/nnet3/run_tdnn.sh and local/chain/run_tdnn.sh (and may eventually
# be called by more scripts).  It contains the common feature preparation and iVector-related parts
# of the script.  See those scripts for examples of usage.


stage=6
nj=30

train_set=train_cleaned   # you might set this to e.g. train.
gmm=tri3_cleaned          # This specifies a GMM-dir from the features of the type you're training the system on;
                          # it should contain alignments for 'train_set'.
online_cmvn_iextractor=false

num_threads_ubm=8
nnet3_affix=_cleaned     # affix for exp/nnet3 directory to put iVector stuff in, so it
                         # becomes exp/nnet3_cleaned or whatever.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp

for f in data/${train_set}/feats.scp ${gmm_dir}/final.mdl; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


# lowres features, alignments
# if [ -f data/${train_set}_sp/feats.scp ] && [ $stage -le 2 ]; then
#   echo "$0: data/${train_set}_sp/feats.scp already exists.  Refusing to overwrite the features "
#   echo " to avoid wasting time.  Please remove the file and continue if you really mean this."
#   exit 1;
# fi

if [ $stage -le 1 ]; then
  echo "$0: stage 1: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh \
    data/${train_set} data/${train_set}_sp

  for datadir in ${train_set}_sp dev test; do
    utils/copy_data_dir.sh data/$datadir data/${datadir}_hires
  done
  echo "run ivector stage 1 done"
fi

if [ $stage -le 2 ]; then
  echo "$0: stage 2: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc_pitch.sh --nj $nj \
    --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "$0: stage 2: fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
  echo "run ivector stage 2 done"
fi

if [ $stage -le 3 ]; then
  if [ -f $ali_dir/ali.1.gz ]; then
    echo "$0: stage 3: alignments in $ali_dir appear to already exist.  Please either remove them "
    echo " stage 3: ... or use a later --stage option."
    exit 1
  fi
  echo "$0: stage 3: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
         data/${train_set}_sp data/lang $gmm_dir $ali_dir
  echo "run ivector stage 3 done"
fi


if [ $stage -le 5 ] && [ -f data/${train_set}_sp_hires/feats.scp ]; then
  echo "$0: stage 4/5 data/${train_set}_sp_hires/feats.scp already exists."
  echo " stage 4/5... Please either remove it, or rerun this script with stage > 2."
  exit 1
fi

if [ $stage -le 5 ]; then
  echo "$0: stage 5 creating high-resolution MFCC features"

  # this shows how you can split across multiple file-systems.  we'll split the
  # MFCC dir across multiple locations.  You might want to be careful here, if you
  # have multiple copies of Kaldi checked out and run the same recipe, not to let
  # them overwrite each other.
  mfccdir=data/${train_set}_sp_hires/data
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $mfccdir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/mfcc/tedlium-$(date +'%m_%d_%H_%M')/s5/$mfccdir/storage $mfccdir/storage
  fi
  echo "run ivector stage 5 done with split dir"
  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires
  echo "run ivector stage 5 done with perturb data dir volume"

  for datadir in ${train_set}_sp dev test; do
    echo "run ivector stage 5 making mfcc start"
    steps/make_mfcc_pitch.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${datadir}_hires
    steps/compute_cmvn_stats.sh data/${datadir}_hires
    utils/fix_data_dir.sh data/${datadir}_hires
    echo "run ivector stage 5 making mfcc done"
  done
  echo "run ivector stage 5 fully done"
fi

if [ $stage -le 6 ]; then
  echo "$0: stage 6 computing a subset of data to train the diagonal UBM."

  mkdir -p exp/nnet3${nnet3_affix}/diag_ubm
  temp_data_root=exp/nnet3${nnet3_affix}/diag_ubm

  # train a diagonal UBM using a subset of about a quarter of the data
  num_utts_total=$(wc -l <data/${train_set}_sp_hires/utt2spk)
  num_utts=$[$num_utts_total/4]
  utils/data/subset_data_dir.sh data/${train_set}_sp_hires \
    $num_utts ${temp_data_root}/${train_set}_sp_hires_subset
  echo "$0: stage 6 - done with subset data dir"
  echo "$0: stage 6 computing a PCA transform from the hires data."
  steps/online/nnet2/get_pca_transform.sh --cmd "$train_cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    ${temp_data_root}/${train_set}_sp_hires_subset \
    exp/nnet3${nnet3_affix}/pca_transform

  echo "$0: stage 6 training the diagonal UBM."
  # Use 512 Gaussians in the UBM.
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 30 \
    --num-frames 700000 \
    --num-threads $num_threads_ubm \
    ${temp_data_root}/${train_set}_sp_hires_subset 512 \
    exp/nnet3${nnet3_affix}/pca_transform exp/nnet3${nnet3_affix}/diag_ubm
  echo "run ivector common: done with stage 6"
fi

if [ $stage -le 7 ]; then
  # Train the iVector extractor. µUse all of the speed-perturbed data since iVector extractors
  # can be sensitive to the amount of data. The script defaults to an iVector dimension of 100.
  echo "$0: stage 7 training the iVector extractor"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 15 \
    --num-threads 4 --num-processes 2 \
    --online-cmvn-iextractor $online_cmvn_iextractor \
    data/${train_set}_sp_hires exp/nnet3${nnet3_affix}/diag_ubm \
    exp/nnet3${nnet3_affix}/extractor || exit 1;
  echo "done with ivector stage 7"
fi

if [ $stage -le 8 ]; then
  echo "run ivector stage 8 start"
  # note, we don't encode the 'max2' in the name of the ivectordir even though
  # that's the data we extract the ivectors from, as it's still going to be
  # valid for the non-'max2' data, the utterance list is the same.
  ivectordir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $ivectordir/storage ]; then
    utils/create_split_dir.pl /export/b0{5,6,7,8}/$USER/kaldi-data/ivectors/tedlium-$(date +'%m_%d_%H_%M')/s5/$ivectordir/storage $ivectordir/storage
  fi
  # We now extract iVectors on the speed-perturbed training data .  With
  # --utts-per-spk-max 2, the script pairs the utterances into twos, and treats
  # each of these pairs as one speaker; this gives more diversity in iVectors..
  # Note that these are extracted 'online' (they vary within the utterance).
  echo "run ivector stage 8 done with split dir"

  # Having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (the iVector starts at zero at the beginning
  # of each pseudo-speaker).
  temp_data_root=${ivectordir}
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${train_set}_sp_hires ${temp_data_root}/${train_set}_sp_hires_max2
  
  echo "run ivector stage 8 done with modifying speaker info"

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
    ${temp_data_root}/${train_set}_sp_hires_max2 \
    exp/nnet3${nnet3_affix}/extractor $ivectordir

  echo "run ivector stage 8 done with extracting ivectors online"

  # Also extract iVectors for the test data, but in this case we don't need the speed
  # perturbation (sp) or small-segment concatenation (comb).
  for data in dev test; do
    echo "run ivector stage 8 start with extracting ivectors online"
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj "$nj" \
      data/${data}_hires exp/nnet3${nnet3_affix}/extractor \
      exp/nnet3${nnet3_affix}/ivectors_${data}_hires
    
    echo "run ivector stage 8 done with extracting ivectors online"
  done
  echo "run ivector stage 8 done fully"
fi

echo "done with run_ivector_common fully"
exit 0;

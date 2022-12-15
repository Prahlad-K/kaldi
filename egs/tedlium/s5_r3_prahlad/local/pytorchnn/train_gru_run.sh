. ./../../cmd.sh
. ./../../path.sh

nj=8
echo "$0: computing neural LM scores of the minimal list of hypotheses."
$train_cmd JOB=1:$nj log/train_gru.JOB.log \
PYTHONPATH=steps/pytorchnn python3 train_gru.py
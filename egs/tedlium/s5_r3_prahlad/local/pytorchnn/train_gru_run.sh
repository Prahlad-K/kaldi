. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$train_cmd
nj=8
echo "$0: training gcnn."
$train_cmd JOB=1:$nj log/train_gru.JOB.log python3 local/pytorchnn/train_gru.py
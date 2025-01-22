#!/bin/bash
set -x

meta_lst=$1
output_dir=$2
lang=$3

timestamp=$(date +%s)
temp_dir=/tmp/wer_calculation_$timestamp
mkdir -p $temp_dir

wav_wav_text=$temp_dir/wav_res_ref_text
score_file=$output_dir/wav_res_ref_text.wer

workdir=$(cd $(dirname $0); cd ../; pwd)

python3 get_wav_res_ref_text.py $meta_lst $output_dir $wav_wav_text
python3 prepare_ckpt.py

num_job=${ARNOLD_WORKER_GPU:-1}
num=$(wc -l $wav_wav_text | awk '{print $1}')
num_per_thread=$((num / num_job + 1))

split -l $num_per_thread --additional-suffix=.lst -d $wav_wav_text $temp_dir/thread-
out_dir=$temp_dir/results/
mkdir -p $out_dir

num_job_minus_1=$((num_job - 1))
if [ $num_job_minus_1 -ge 0 ]; then
    for rank in $(seq 0 $num_job_minus_1); do
        sub_score_file=$out_dir/thread-0${rank}.wer.out
        CUDA_VISIBLE_DEVICES=$rank python3 run_wer.py $temp_dir/thread-0${rank}.lst $sub_score_file $lang &
    done
fi
wait

cat $out_dir/thread-0*.wer.out > $out_dir/merge.out
python3 /home/renkow/seed-tts-eval/average_wer.py $out_dir/merge.out $score_file

rm -rf $temp_dir

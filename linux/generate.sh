#!/bin/sh

work_dir=$(cd "$(dirname "$0")"/..;pwd)

python3 $work_dir/scripts/generate.py \
	--input_text='望着天空的月亮' \
	--length=128 \
	--nsamples=1 \
	--vocab_path="$work_dir/pre_data/weixiao/vocab.txt" \
	--model_path="$work_dir/trained_model/final_model" \
#	--is_slow_model=True

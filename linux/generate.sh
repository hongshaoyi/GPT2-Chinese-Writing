#!/bin/sh

python3 ./scripts/generate.py \
	--input_text='望着天空的月亮' \
	--length=128 \
	--nsamples=1 \
	--vocab_path='pre_data/faguo/vocab.txt' \
	--model_path='trained_model/final_model' \
	--is_slow_model=True

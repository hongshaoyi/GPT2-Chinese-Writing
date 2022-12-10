#!/bin/sh

work_dir=$(cd "$(dirname "$0")"/..;pwd)
dir_name=$1

if [ ! -n "$dir_name" ]; then
	echo "请指定要训练pre_data下的哪个预处理的语料目录！"

	return
fi

python3 $work_dir/scripts/train.py \
	--work_dir $work_dir \
	--dir_name $dir_name \
	--epochs=1 \
	--batch_size=16 \
	--log_step=1 \
	--stride=128 \
	--gradient_accumulation=1 \
#	--pretrained_model='trained_model/final_model/'

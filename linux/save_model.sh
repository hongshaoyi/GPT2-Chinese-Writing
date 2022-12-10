#!/bin/sh

work_dir=$(cd "$(dirname "$0")"/..;pwd)
dir_name=$1

if [ ! -n "$dir_name" ]; then
	echo "请指定模型保存目录！"

	return
fi

if [ -d "$work_dir/model/$dir_name" ]; then
  	rm -r $work_dir/model/$dir_name

  	if [ $? -ne 0 ]; then
		return
	fi
fi

cp -r $work_dir/trained_model/final_model $work_dir/model/$dir_name
cp $work_dir/pre_data/$dir_name/vocab.txt $work_dir/model/$dir_name/vocab.txt


#!/bin/sh

work_dir=$(cd "$(dirname "$0")"/..;pwd)
dir_name=$1

if [ ! -n "$dir_name" ]; then
	echo "请指定训练语料放在data下的哪个目录下！"

	return
fi

if [ ! -d "$work_dir/pre_data/$dir_name" ]; then
  	mkdir $work_dir/pre_data/$dir_name

  	if [ $? -ne 0 ]; then
		return
	fi
fi

python3 $work_dir/scripts/build_pretrain_file.py --work_dir $work_dir --dir_name $dir_name

if [ $? -ne 0 ]; then
	return
fi

echo "训练语料已处理完毕,生成文件放在pre_data/$dir_name目录下"

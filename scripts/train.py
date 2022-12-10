import transformers
import torch
import os
import math 
import random
import argparse
import numpy as np
from datetime import datetime
from torch.nn import DataParallel
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--work_dir', type=str, required=True, help='工作目录')
    parser.add_argument('--dir_name', type=str, required=True, help='存放目录')
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='设置使用哪些显卡')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练循环')
    parser.add_argument('--batch_size', default=8, type=int, required=False, help='训练batch size')
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False, help='学习率')
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False, help='warm up步数')
    parser.add_argument('--log_step', default=1, type=int, required=False, help='多少步汇报一次loss')
    parser.add_argument('--stride', default=768, type=int, required=False, help='训练时取训练数据的窗口步长')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False, help='梯度积累')
    parser.add_argument('--fp16', action='store_true', help='混合精度')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='模型训练起点路径')
 
    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    work_dir = args.work_dir
    dir_name = args.dir_name
    pre_data_dir = work_dir + '/pre_data/' + dir_name
    pre_file_path = pre_data_dir + '/pre_train.txt'
    model_config_path = pre_data_dir + '/model_config.json'
    output_dir = work_dir + '/trained_model'

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(model_config_path)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)

    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16  # 不支持半精度的显卡请勿打开
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        pretrained_model_dir = work_dir + '/' + args.pretrained_model

        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(pretrained_model_dir)
    
    model.train()
    model.to(device)
    
    full_len = 0
    token_list = []

    with open(pre_file_path, 'r') as f:
        print("load file: ", pre_file_path)

        for token in (f.read().strip().split()):
            token_list.append(int(token))

        full_len = len(token_list)
    
    print("load finish!")

    start_point = 0
    samples = []
    
    while start_point < full_len - n_ctx:
        samples.append(token_list[start_point:start_point + n_ctx])
        start_point += stride

    #这里保证了samples里的每个元素长度一致
    #这样在下面的batch size矩阵不会报错
    #因为矩阵要求每行必须长度一致
    if start_point < full_len:
        samples.append(token_list[full_len - n_ctx:])
    
    sample_size = len(samples)
    print("samples size: ", sample_size)
     
    total_steps = math.ceil(math.ceil(sample_size / batch_size) / gradient_accumulation) * epochs
    print('total steps: ', total_steps)

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
#    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,t_total=total_steps)
    
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    multi_gpu = False
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        model = DataParallel(model)
        multi_gpu = True
    
    running_loss = 0
    model_to_save = model.module if hasattr(model, 'module') else model
    
    print('starting training')
    
    for epoch in range(epochs):
        now = datetime.now()
        
        print('epoch: ', epoch + 1)
        print('now time: ', now)
         
        random.shuffle(samples)
        
        for step in range(math.ceil(sample_size / batch_size)):
            #  prepare data
            end = min((step + 1) * batch_size, sample_size)
            batch = samples[step * batch_size:end]
            batch_labels = []
            batch_inputs = []
            
            for ids in batch:
                batch_labels.append(ids)
                batch_inputs.append(ids)
            
            batch_labels = torch.tensor(batch_labels).long().to(device)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)
            
            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_labels)
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:
                loss = loss.mean()

            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (step + 1) % gradient_accumulation == 0 or end == sample_size: 
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
#                scheduler.step()

            if (step + 1) % log_step == 0 or end == sample_size:
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    (step + 1) // gradient_accumulation,
                    epoch + 1,
                    running_loss / log_step))

                running_loss = 0
 
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: ', then)
        print('time for one epoch: ', then - now)

        #最后一次就不用存了,会存在final model里
        if epoch + 1 < epochs:
            print('saving model for epoch {}'.format(epoch + 1))
        
            if not os.path.exists(output_dir + '/model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + '/model_epoch{}'.format(epoch + 1))
        
            model_to_save.save_pretrained(output_dir + '/model_epoch{}'.format(epoch + 1))

    print('training finished')
    
    if not os.path.exists(output_dir + '/final_model'):
        os.mkdir(output_dir + '/final_model')
    
    model_to_save.save_pretrained(output_dir + '/final_model')

if __name__ == '__main__':
    main()

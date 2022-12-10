import torch
import torch.nn.functional as F
import os
import argparse
import thulac
from tqdm import trange
from transformers import GPT2LMHeadModel


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_text', type=str, required=True, help='生成文章的开头')
    parser.add_argument('--length', type=int, required=True, help='生成长度')
    parser.add_argument('--nsamples', type=int, required=True, help='生成几个样本')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
    parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
    parser.add_argument('--save_file_name', default='samples.txt', type=str, required=False, help="保存样本的文件名")
    parser.add_argument('--is_slow_model', default=False, type=bool, required=False, help="是否使用慢速模式,默认是快速")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    
    length = args.length
    nsamples = args.nsamples
    temperature = args.temperature
    top_k = args.topk
    top_p = args.topp

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    n_ctx = model.config.n_ctx

    if args.save_samples_path:
        if not os.path.exists(args.save_samples_path):
            os.makedirs(args.save_samples_path)
        samples_file = open(args.save_samples_path + '/' + args.save_file_name, 'w', encoding='utf8')
   
    vocab_file_path = args.vocab_path
    vocab_dict = {}
    text_dict = {}

    with open(vocab_file_path, "r") as f:
        print("load vocab file!")
        lines = f.readlines()

        for i, line in enumerate(lines):
            vocab = str(line.strip())
            vocab_dict[vocab] = i
            text_dict[i] = vocab


        print("load vocab finish!")

    input_text = args.input_text
    lac = thulac.thulac(seg_only=True)
    cut_text = lac.cut(args.input_text, text=True)
    text_list = cut_text.strip().split()
    context_tokens = []

    for word in text_list:
        token = 0

        if word in vocab_dict:
            token = vocab_dict[word]
        else:
            token = vocab_dict['[UNK]']

        context_tokens.append(token)

    is_slow_model = args.is_slow_model
    past = None
    prev = None

    if not is_slow_model:
        inputs = torch.LongTensor(context_tokens).view(1, -1).to(device)
    
        if len(context_tokens) > 1:
            _, past = model(inputs[:, :-1], None)[:2]
            prev = inputs[:, -1].view(1, -1)
        else:
            past = None
            prev = inputs
    
        pre_length = len(context_tokens)

    for index in range(nsamples): 
        generate_text_list = []
        now_past = None
        now_prev = None
        input_list = None

        if is_slow_model:
            input_list = [] + context_tokens
        else:
            now_past = past
            now_prev = prev

        with torch.no_grad():
            for i in trange(length):
                now_output = None

                if is_slow_model:
                    inputs_ids = {'input_ids': torch.tensor([input_list[-(n_ctx - 1):]])}
                    now_output = model(**inputs_ids)
                    now_output = now_output[0][0, -1, :]
                    now_output = now_output / temperature
                else:
                    if i + pre_length - 1 > n_ctx - 1:
                        new_past = ()
                        
                        for index, sub_past in enumerate(now_past):
                            drop_past, split_past = torch.split(sub_past, split_size_or_sections=[1, n_ctx - 1], dim = 3)
                            new_past += (split_past,)

                        now_past = new_past

                    now_output = model(now_prev, past=now_past)
                    now_output, now_past = now_output[:2]
                    now_output = now_output[-1].squeeze(0) / temperature

                filtered_logits = top_k_top_p_filtering(now_output, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
                token = next_token.item()
                text = text_dict[token]
             
                if text == '[SEP]':
                    text = '\n'
             
                generate_text_list.append(text)

                if is_slow_model:
                    input_list.append(token)
                else:
                    now_prev = next_token.view(1, 1)

        final_text = input_text + ''.join(generate_text_list)
        
        info = "=" * 40 + " SAMPLE " + str(index+1) + " " + "=" * 40 + "\n"
        print(info)
        print(final_text)
        
        if args.save_samples_path:
            samples_file.write(info)
            samples_file.write(final_text)
            samples_file.write('\n')
            samples_file.write('=' * 80)
            samples_file.write('\n' * 2)

    print("=" * 80)
    if args.save_samples_path:
       samples_file.close()


if __name__ == '__main__':
    main()

import os
import argparse
import thulac
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--work_dir', type=str, required=True, help='工作目录')
    parser.add_argument('--dir_name', type=str, required=True, help='存放目录')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())
    
    work_dir = args.work_dir
    dir_name = args.dir_name
    train_dir = work_dir + '/data/' + dir_name
    output_dir = work_dir + '/pre_data/' + dir_name
    vocab_file_path = output_dir + '/vocab.txt'
    pre_file_path = output_dir + '/pre_train.txt'
    template_model_confg_path = work_dir + '/config/model_config_template.json'
    model_config_path = output_dir + '/model_config.json'

    vocab_dict = {'[SEP]': 0, '[CLS]': 1, '[MASK]': 2, '[PAD]': 3, '[UNK]': 4}
    vocab_index = len(vocab_dict)
    lac = thulac.thulac(seg_only=True)
    save_lines = []

    for path, dir_list, file_list in os.walk(train_dir):
        for file_name in file_list:
            file_path = os.path.join(path, file_name)

            with open(file_path, 'r', encoding='utf8') as f:
                print("deal file:", file_path)
                lines = f.readlines()
                
                for i, line in enumerate(tqdm(lines)):
                    lines_text = lac.cut(line, text=True)
                    text_list = lines_text.strip().split()
                    save_line = []
  
                    for word in text_list:
                        if not word in vocab_dict:
                            vocab_dict[word] = vocab_index
                            vocab_index += 1
  
                        save_lines.append(str(vocab_dict[word]))
  
                    save_lines.append(str(vocab_dict['[SEP]']))

                print("deal finsih!")

    print("all files deal over!")

    vocab_num = len(vocab_dict)
    
    with open(vocab_file_path, "w") as f:
        print("save vocab file, total vocab num: ", vocab_num)

        f.write('\n'.join(vocab_dict))

        print("save success!")

    with open(pre_file_path, "w") as f:
        print("save pre train file, total words: ", len(save_lines))

        f.write(" ".join(save_lines))

        print("save success!")

    with open(template_model_confg_path, "r") as f:
        model_config_json = json.load(f)
        model_config_json['vocab_size'] = vocab_num

        with open(model_config_path, "w") as write_f:
            print("save model config file!")
            print(json.dumps(model_config_json, indent=4))

            json.dump(model_config_json, write_f, indent=4)

            print("save success!")

    print("all done!")

if __name__ == "__main__":
    main()


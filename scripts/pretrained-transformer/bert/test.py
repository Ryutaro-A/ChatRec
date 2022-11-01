
import sys
import os
import json
import argparse
from socket import gethostname

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import BertForSequenceClassification

sys.path.append("transformer")
from mydataset import MyDataset
from myutils import *


def test_save(args, split_id, model_output_dir, result_output_dir):
    max_len = args.max_len

    print(f'model_output_dir: {model_output_dir}')
    print(f'result_output_dir: {result_output_dir}')

    if not os.path.isdir(result_output_dir):
        os.makedirs(result_output_dir)


    # dataset, dataloader
    test_dataset = MyDataset(args.data_dir, args.split_info_file, split_id, "test", args.down_scale, max_len, args.bert_vocab_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    # GPU
    device_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
    os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model
    config = transformers.BertConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        model_type="bert",
        num_attention_heads=12,
        num_hidden_layers=12,
        pad_token_id=0,
        tokenizer_class="BertJapaneseTokenizer",
        type_vocab_size=2,
        vocab_size=32000
    )
    setattr(config, "num_labels", 1)

    model = transformers.BertForSequenceClassification.from_pretrained(
            "cl-tohoku/bert-base-japanese-whole-word-masking",
            config=config
            )
    model = nn.DataParallel(model, device_ids=[0,1]).to(device)
    model.load_state_dict(torch.load(model_output_dir+"best_loss_model-"+gethostname()+".pt"))
    #model = nn.DataParallel(model).to(device)


    # test
    model.eval()
    result_dict = {}
    for data_list in test_loader:
        filename = data_list[0]["filename"][0]
        speaker = data_list[0]["speaker"][0]
        if filename not in result_dict:
            result_dict[filename] = {}
        result_dict[filename][speaker] = []

        with torch.no_grad():
            src, _, spot_id_list = stack_data(data_list)
            pred = get_pred(src, model, device)

        for pred_score, spot_id in zip(pred, spot_id_list):
            tmp_dict = {"score": pred_score.item(), "id": spot_id}
            result_dict[filename][speaker].append(tmp_dict)


    # 結果書き出し
    for filename, result_jd in result_dict.items():
        output_filename = filename.replace(".json", ".rmd.json")
        tmp_dir = result_output_dir + output_filename
        tmp_dir = tmp_dir.split('/')[:-1]
        if not os.path.isdir("/".join(tmp_dir)):
            os.makedirs("/".join(tmp_dir))
        with open(result_output_dir + output_filename, mode="w") as out_f:
            json.dump(result_jd, out_f, indent=4)



# parser = argparse.ArgumentParser()

# parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
# parser.add_argument("--split_info_file", help="分割方法のjsonファイル")
# parser.add_argument("--split_id", type=int, help="分割の何番目を使用するか(0~9)")
# parser.add_argument("--saved_model_dir", help="モデルが保存されたディレクトリ")
# parser.add_argument("--result_output_dir", help="結果を出力するディレクトリ")

# parser.add_argument("--down_scale", type=bool, default=False, help="スコアのダウンスケールをするかどうか")
# parser.add_argument("--bert_vocab_dir", help="tokenizerのvocabファイルのディレクトリ")

# parser.add_argument("--max_len", type=int, default=1024)
# parser.add_argument("--vocab_size", type=int, default=32000)
# parser.add_argument("--hidden_size", type=int, default=768)
# parser.add_argument("--num_layers", type=int, default=4)
# parser.add_argument("--num_heads", type=int, default=8)
# parser.add_argument("--intermadiate_size", type=int, default=2048)

# args = parser.parse_args()


# max_len = args.max_len

# if not os.path.isdir(args.result_output_dir):
#     os.makedirs(args.result_output_dir)


# # dataset, dataloader
# test_dataset = MyDataset(args.data_dir, args.split_info_file, "test", args.down_scale, max_len, args.bert_vocab_dir)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# # GPU
# device_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
# os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # model
# config = transformers.BertConfig(
#     attention_probs_dropout_prob=0.1,
#     hidden_act="gelu",
#     hidden_dropout_prob=0.1,
#     hidden_size=768,
#     initializer_range=0.02,
#     intermediate_size=3072,
#     layer_norm_eps=1e-12,
#     max_position_embeddings=512,
#     model_type="bert",
#     num_attention_heads=12,
#     num_hidden_layers=12,
#     pad_token_id=0,
#     tokenizer_class="BertJapaneseTokenizer",
#     type_vocab_size=2,
#     vocab_size=32000
#     )
# setattr(config, "num_labels", 1)

# model = transformers.BertForSequenceClassification.from_pretrained(
#         "cl-tohoku/bert-base-japanese-whole-word-masking",
#         config=config
#         )
# model = nn.DataParallel(model, device_ids=[0,1]).to(device)
# model.load_state_dict(torch.load(args.saved_model_dir+"best_loss_model-"+gethostname()+".pt"))
# #model = nn.DataParallel(model).to(device)


# # test
# model.eval()
# result_dict = {}
# for data_list in test_loader:
#     filename = data_list[0]["filename"][0]
#     speaker = data_list[0]["speaker"][0]
#     if filename not in result_dict:
#         result_dict[filename] = {}
#     result_dict[filename][speaker] = []

#     with torch.no_grad():
#         src, _, spot_id_list = stack_data(data_list)
#         pred = get_pred(src, model, device)

#     for pred_score, spot_id in zip(pred, spot_id_list):
#         tmp_dict = {"score": pred_score.item(), "id": spot_id}
#         result_dict[filename][speaker].append(tmp_dict)


# # 結果書き出し
# for filename, result_jd in result_dict.items():
#     output_filename = filename.replace(".json", ".rmd.json")
#     tmp_dir = args.result_output_dir + output_filename
#     tmp_dir = tmp_dir.split('/')[:-1]
#     if not os.path.isdir("/".join(tmp_dir)):
#         os.makedirs("/".join(tmp_dir))
#     with open(args.result_output_dir + output_filename, mode="w") as out_f:
#         json.dump(result_jd, out_f, indent=4)
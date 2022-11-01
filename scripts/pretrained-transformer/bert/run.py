import argparse
import sys

sys.path.append("scripts")
from train import train_valid
from test import test_save
from ndcg2 import ndcg_all
from recall_at_k import recall_all
import time

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
parser.add_argument("--data_type", help="データの種類")
parser.add_argument("--split_info_file", help="分割方法のjsonファイル")

parser.add_argument("--bert_vocab_dir", help="tokenizerのvocabファイルのディレクトリ")
parser.add_argument("--down_scale", action='store_true')
parser.add_argument("--use_pretrain_model", action='store_true')
parser.add_argument("--use_cuda", action='store_true')
parser.add_argument("--use_large_model", action='store_true')
parser.add_argument("--use_device_ids", type=str, default="all")

parser.add_argument("--max_len", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_epoch", type=int, default=100)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--bert_hidden_dropout", type=float, default=0.0)
parser.add_argument("--bert_attention_dropout", type=float, default=0.0)
parser.add_argument("--optimizer", type=str, default=0.0)

parser.add_argument("--start_id", type=int, default=0)
parser.add_argument("--end_id", type=int, default=0)

args = parser.parse_args()

with open(f'./transformer/result_{args.data_type}_{args.start_id}-{args.end_id}.txt', mode='w', encoding='utf-8') as f:
    for split_id in range(args.start_id, args.end_id+1):
        model_output_dir = f'./transformer/saved_model/bert/{split_id}/{args.data_type}_split/'
        result_output_dir = f'./transformer/result/{split_id}/{args.data_type}_split/'
        train_valid(args, split_id, model_output_dir, result_output_dir)

        time.sleep(120)

        test_save(args, split_id, model_output_dir, result_output_dir)

        time.sleep(120)

        ndcg_str = ndcg_all(result_output_dir+args.data_type+'/', args.data_dir+args.data_type+'/')
        recall_str = recall_all(result_output_dir+args.data_type+'/', args.data_dir+args.data_type+'/')

        f.write(f'{args.data_type}: {split_id}\n{ndcg_str} {recall_str}\n')
        f.flush()

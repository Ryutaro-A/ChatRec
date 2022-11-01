import os
import time
from socket import gethostname
import sys
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import transformers
from transformers import BertForSequenceClassification

sys.path.append("pretrained-transformer/bert/util")
from dataset import MyDataset
from myutils import *
from ndcg import ndcg_dict

def test_save(args, model, test_loader, device):
    # test
    model.eval()
    result_dict = {}
    for i, data_list in enumerate(test_loader):
        filename = data_list[0]["filename"][0]
        speaker = data_list[0]["speaker"][0]
        if filename not in result_dict:
            result_dict[filename] = {}
        result_dict[filename][speaker] = []

        with torch.no_grad():
            src, tgt, spot_id_list = stack_data(data_list)
            pred = get_pred(src, model, device)

        # dic = ndcg_dict(pred.cpu().numpy().tolist(), tgt.cpu().numpy().tolist(), 10)

        for pred_score, spot_id in zip(pred, spot_id_list):
            tmp_dict = {"score": pred_score.item(), "id": spot_id}
            result_dict[filename][speaker].append(tmp_dict)

    # 結果書き出し
    for filename, result_jd in result_dict.items():
        output_filename = filename.replace(".json", ".rmd.json")
        tmp_dir = args.output_dir + output_filename
        tmp_dir = tmp_dir.split('/')[:-1]
        if not os.path.isdir("/".join(tmp_dir)):
            os.makedirs("/".join(tmp_dir))

        with open(args.output_dir + output_filename, mode="w") as out_f:
            json.dump(result_jd, out_f, indent=4)

def evaluation(valid_loader, model, criterion, device_):
    model.eval()
    total_loss, total_ndcg = 0, 0
    with torch.no_grad():
        for data_list in valid_loader:
            # 一度にモデルに流すためにデータを結合
            src, tgt, _ = stack_data(data_list)
            tgt = tgt.float().to(device_)
            pred = get_pred(src, model, device_)
            loss = criterion(pred, tgt)
            total_loss += loss.item()
            ndcg_tgt = tgt.cpu()*2 + 3
            ndcg = ndcg_dict(pred.cpu(), ndcg_tgt)
            total_ndcg += ndcg[1]
    return total_loss, total_ndcg



def train(train_loader, model, optimizer, criterion, device_):
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        optimizer.zero_grad()
        tgt = tgt.float().to(device_)
        pred = get_pred(src, model, device_)
        loss = criterion(pred, tgt)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss

def run(args):

    split_info_file = f'{args.split_info_dir}{args.data_type}_split.json'

    print(f"PID: {os.getpid()}, Server: {gethostname()}")
    print(f'method: {args.method}')
    print(f"split id: {args.split_id}")
    print(f"split file: {split_info_file}")
    print(f"batch size: {args.batch_size}")
    print(f"max len: {args.max_len}")
    print(f"data dir: {args.data_dir}")
    print(f"model output dir: {args.model_output_dir}")
    print(f"bert vocab dir: {args.vocab_dir}")

    # モデル保存用ディレクトリ作成
    if not os.path.isdir(args.model_output_dir):
        os.makedirs(args.model_output_dir)

    # dataset, dataloader
    train_dataset = MyDataset(args.data_dir, split_info_file, args.split_id, "train", False, args.max_len, args.vocab_dir)
    valid_dataset = MyDataset(args.data_dir, split_info_file, args.split_id, "dev", False, args.max_len, args.vocab_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # model
    config = transformers.BertConfig(
        attention_probs_dropout_prob=args.attention_dropout,
        hidden_act="gelu",
        hidden_dropout_prob=args.hidden_dropout,
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

    # model
    if args.use_pretrain_model:
        model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", config=config)
    else:
        model = BertForSequenceClassification(config)

    # GPU
    if args.use_cuda:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("You don't have GPU! Run CPU mode.")
        else:
            if args.use_device_ids == "all":
                use_device_id = [id_ for id_ in range(torch.cuda.device_count())]
            else:
                use_device_id = [int(id_) for id_ in args.use_device_ids]
                device = 'cuda:'+str(use_device_id[0])
            model = nn.DataParallel(model, device_ids=use_device_id).to(device)
            print(f'You can use device id -> {use_device_id}')
    else:
        device = 'cpu'

    # optim
    if args.optimizer == "Adafactor":
        optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, relative_step=False)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # loss function
    criterion = nn.MSELoss(reduction="mean")

    best_valid_ndcg, best_epoch = 0, 0
    best_valid_loss, best_loss_epoch = 100000, 0
    tmp_patience = 0
    len_loader_train, len_loader_valid = len(train_loader), len(valid_loader)

    for epoch in range(1, args.max_epoch+1):
        start_time = time.time()
        train_loss = train(train_loader, model, optimizer, criterion, device)
        valid_loss, valid_ndcg = evaluation(valid_loader, model, criterion, device)
        train_loss /= len_loader_train
        valid_loss /= len_loader_valid
        valid_ndcg /= len_loader_valid

        print(f"\nEpoch[{epoch}/{args.max_epoch}]")
        print(f"<Loss> train: {train_loss}, valid: {valid_loss}")
        print(f"<Valid NDCG> {valid_ndcg}, <Time> {time.time()-start_time}")

        if valid_ndcg > best_valid_ndcg:
            best_valid_ndcg, best_epoch = valid_ndcg, epoch
            torch.save(model.module.state_dict(), f"{args.model_output_dir}best_ndcg_model.pt")
            torch.save(optimizer.state_dict(), f"{args.model_output_dir}best_ndcg_optim.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss, best_loss_epoch = valid_loss, epoch
            tmp_patience = 0
            torch.save(model.state_dict(), f"{args.model_output_dir}best_loss_model.pt")
            torch.save(optimizer.state_dict(), f"{args.model_output_dir}best_loss_optim.pt")
        else:
            tmp_patience += 1
            if tmp_patience > args.patience:
                break

        print(f"<Best> NDCG={best_valid_ndcg}, NDCG epoch={best_epoch}, Loss={best_valid_loss}, loss epoch={best_loss_epoch}")


    # 結果保存用ディレクトリの作成
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # dataset, dataloader
    test_dataset = MyDataset(args.data_dir, split_info_file, args.split_id, "test", False, args.max_len, args.vocab_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    test_save(args, model, test_loader, device)
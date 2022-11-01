import sys
import os
import time
from socket import gethostname

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter

import transformers
from transformers import BertForSequenceClassification

sys.path.append("scripts")
from mydataset import MyDataset
from myutils import *
from ndcg import ndcg_dict

def train_valid(args, split_id, model_output_dir, result_output_dir):
    print(f"PID: {os.getpid()}, Server: {gethostname()}")


    print(f"split file: {args.split_info_file}")
    print(f"split id: {split_id}")
    print(f"batch size: {args.batch_size}")
    print(f'model_output_dir: {model_output_dir}')
    print(f'result_output_dir: {result_output_dir}')

    max_len = args.max_len

    # モデル保存用ディレクトリ作成
    if not os.path.isdir(model_output_dir):
        os.makedirs(model_output_dir)


    # dataset, dataloader
    train_dataset = MyDataset(args.data_dir, args.split_info_file, split_id, "train", args.down_scale, max_len, args.bert_vocab_dir)
    valid_dataset = MyDataset(args.data_dir, args.split_info_file, split_id, "dev", args.down_scale, max_len, args.bert_vocab_dir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

    # model
    config = transformers.BertConfig(
        attention_probs_dropout_prob=args.bert_attention_dropout,
        hidden_act="gelu",
        hidden_dropout_prob=args.bert_hidden_dropout,
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

    # loss function
    criterion = nn.MSELoss(reduction="mean")

    # model]
    if args.use_large_model:
        if args.use_pretrain_model:
            model = transformers.BertForSequenceClassification.from_pretrained(
                "cl-tohoku/bert-base-japanese-whole-word-masking",
                config=config
            )
        else:
            # model = transformers.BertForSequenceClassification(large_config)
            pass
    else:
        if args.use_pretrain_model:
            model = transformers.BertForSequenceClassification.from_pretrained(
                "cl-tohoku/bert-base-japanese-whole-word-masking",
                config=config
            )
        else:
            model = transformers.BertForSequenceClassification(config)

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



    def evaluation(valid_loader, model, device_):
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
            #print(pred.size())
            loss = criterion(pred, tgt)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        return total_loss

    best_valid_ndcg, best_epoch = 0, 0
    best_valid_loss, best_loss_epoch = 100000, 0
    tmp_patience = 0
    len_loader_train, len_loader_valid = len(train_loader), len(valid_loader)

    for epoch in range(1, args.max_epoch+1):
        start_time = time.time()
        train_loss = train(train_loader, model, optimizer, criterion, device)
        valid_loss, valid_ndcg = evaluation(valid_loader, model, device)
        train_loss /= len_loader_train
        valid_loss /= len_loader_valid
        valid_ndcg /= len_loader_valid

        print(f"\nEpoch[{epoch}/{args.max_epoch}]")
        print(f"<Loss> train: {train_loss}, valid: {valid_loss}")
        print(f"<Valid NDCG> {valid_ndcg}, <Time> {time.time()-start_time}")

        if valid_ndcg > best_valid_ndcg:
            best_valid_ndcg, best_epoch = valid_ndcg, epoch
            torch.save(model.module.state_dict(), f"{model_output_dir}best_ndcg_model-{gethostname()}.pt")
            torch.save(optimizer.state_dict(), f"{model_output_dir}best_ndcg_optim-{gethostname()}.pt")

        if valid_loss < best_valid_loss:
            best_valid_loss, best_loss_epoch = valid_loss, epoch
            tmp_patience = 0
            torch.save(model.state_dict(), f"{model_output_dir}best_loss_model-{gethostname()}.pt")
            torch.save(optimizer.state_dict(), f"{model_output_dir}best_loss_optim-{gethostname()}.pt")
        else:
            tmp_patience += 1
            if tmp_patience > args.patience:
                break

        print(f"<Best> NDCG={best_valid_ndcg}, NDCG epoch={best_epoch}, Loss={best_valid_loss}, loss epoch={best_loss_epoch}")


# print(f"PID: {os.getpid()}, Server: {gethostname()}")

# model_id = 2


# parser = argparse.ArgumentParser()

# parser.add_argument("--data_dir", help="対話&評価のjsonファイルのディレクトリ")
# parser.add_argument("--split_info_file", help="分割方法のjsonファイル")
# parser.add_argument("--split_id", type=int, help="分割の何番目を使用するか(0~9)")
# parser.add_argument("--model_output_dir", help="モデルを保存するディレクトリ")

# parser.add_argument("--down_scale", type=bool, default=True, help="スコアのダウンスケールをするかどうか")
# parser.add_argument("--bert_vocab_dir", help="tokenizerのvocabファイルのディレクトリ")

# parser.add_argument("--use_pretrained_bert", action='store_true', default=False, help="学習済みBERTを使用するかどうか")

# parser.add_argument("--max_len", type=int, default=1024)
# parser.add_argument("--vocab_size", type=int, default=32000)
# parser.add_argument("--hidden_size", type=int, default=768)
# parser.add_argument("--num_layers", type=int, default=4)
# parser.add_argument("--num_heads", type=int, default=8)
# parser.add_argument("--intermadiate_size", type=int, default=2048)

# parser.add_argument("--batch_size", type=int, default=32)
# parser.add_argument("--max_epoch", type=int, default=100)
# parser.add_argument("--patience", type=int, default=20)
# parser.add_argument("--lr", type=float, default=1e-4)

# args = parser.parse_args()


# print(f"split file: {args.split_info_file}")
# print(f"split id: {args.split_id}")
# print(f"batch size: {args.batch_size}")

# max_len = args.max_len
# model_output_dir = args.model_output_dir

# # モデル保存用ディレクトリ作成
# if not os.path.isdir(model_output_dir):
#     os.makedirs(model_output_dir)


# # dataset, dataloader
# train_dataset = MyDataset(args.data_dir, args.split_info_file, "train", args.down_scale, max_len, args.bert_vocab_dir)
# valid_dataset = MyDataset(args.data_dir, args.split_info_file, "dev", args.down_scale, max_len, args.bert_vocab_dir)
# train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
# valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)


# # GPU
# device_ids = ",".join(str(i) for i in range(torch.cuda.device_count()))
# #os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
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


# # optim
# optimizer = transformers.Adafactor(model.parameters(), lr=args.lr, relative_step=False)
# #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# #optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
# # loss function
# criterion = nn.MSELoss(reduction="mean")



# def evaluation(valid_loader, model, device_):
#     model.eval()
#     total_loss, total_ndcg = 0, 0
#     with torch.no_grad():
#         for data_list in valid_loader:
#             # 一度にモデルに流すためにデータを結合
#             src, tgt, _ = stack_data(data_list)
#             tgt = tgt.float().to(device_)
#             pred = get_pred(src, model, device_)
#             loss = criterion(pred, tgt)
#             total_loss += loss.item()
#             ndcg_tgt = tgt.cpu()*2 + 3
#             ndcg = ndcg_dict(pred.cpu(), ndcg_tgt)
#             total_ndcg += ndcg[1]
#     return total_loss, total_ndcg



# def train(train_loader, model, optimizer, criterion, device_):
#     model.train()
#     total_loss = 0
#     for src, tgt in train_loader:
#         optimizer.zero_grad()
#         tgt = tgt.float().to(device_)
#         pred = get_pred(src, model, device_)
#         #print(pred.size())
#         loss = criterion(pred, tgt)
#         total_loss += loss.item()
#         loss.backward()
#         optimizer.step()
#     return total_loss

# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter(log_dir="./logs")

# best_valid_ndcg, best_epoch = 0, 0
# best_valid_loss, best_loss_epoch = 100000, 0
# tmp_patience = 0
# len_loader_train, len_loader_valid = len(train_loader), len(valid_loader)

# for epoch in range(1, args.max_epoch+1):
#     start_time = time.time()
#     train_loss = train(train_loader, model, optimizer, criterion, device)
#     valid_loss, valid_ndcg = evaluation(valid_loader, model, device)
#     train_loss /= len_loader_train
#     valid_loss /= len_loader_valid
#     valid_ndcg /= len_loader_valid

#     print(f"\nEpoch[{epoch}/{args.max_epoch}]")
#     print(f"<Loss> train: {train_loss}, valid: {valid_loss}")
#     print(f"<Valid NDCG> {valid_ndcg}, <Time> {time.time()-start_time}")

#     writer.add_scalar("Train : Loss/train", train_loss, epoch)
#     writer.add_scalar("Valid : Loss/valid", valid_loss, epoch)

#     if valid_ndcg > best_valid_ndcg:
#         best_valid_ndcg, best_epoch = valid_ndcg, epoch
#         torch.save(model.module.state_dict(), f"{model_output_dir}best_ndcg_model-{gethostname()}.pt")
#         torch.save(optimizer.state_dict(), f"{model_output_dir}best_ndcg_optim-{gethostname()}.pt")

#     if valid_loss < best_valid_loss:
#         best_valid_loss, best_loss_epoch = valid_loss, epoch
#         tmp_patience = 0
#         torch.save(model.state_dict(), f"{model_output_dir}best_loss_model-{gethostname()}.pt")
#         torch.save(optimizer.state_dict(), f"{model_output_dir}best_loss_optim-{gethostname()}.pt")
#     else:
#         tmp_patience += 1
#         if tmp_patience > args.patience:
#             break

#     print(f"<Best> NDCG={best_valid_ndcg}, NDCG epoch={best_epoch}, Loss={best_valid_loss}, loss epoch={best_loss_epoch}")
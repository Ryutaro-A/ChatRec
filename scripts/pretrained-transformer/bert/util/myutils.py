
import itertools
from operator import itemgetter
import torch
from transformers import AutoTokenizer
import numpy as np


def get_pred(src, model, device):
    input_ids = src["input_ids"].to(device)
    token_type_ids = src["token_type_ids"].to(device)
    attention_mask = src["attention_mask"].to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    logits = output["logits"].squeeze()
    pred = torch.tanh(logits)
    return pred
    #return logits


def stack_data(data_list):
    input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
    score_list, spot_id_list = [], []
    for data in data_list:
        #print(data["src"]["input_ids"])
        input_ids_list.append(data["src"]["input_ids"])
        token_type_ids_list.append(data["src"]["token_type_ids"])
        attention_mask_list.append(data["src"]["attention_mask"])
        score_list.append(data["score"])
        spot_id_list.append(data["id"][0])
    input_ids = torch.stack(input_ids_list).squeeze()
    token_type_ids = torch.stack(token_type_ids_list).squeeze()
    attention_mask = torch.stack(attention_mask_list).squeeze()
    src = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": attention_mask}
    tgt = torch.stack(score_list).squeeze()
    return src, tgt, spot_id_list


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result


# 赤くハイライトする
def highlight_r(word, attn):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - attn-0.2)), int(255*(1 - attn-0.2)))
    # html_color = '#%02X%02X%02X' % (255, int(255-(100*attn)), int(255-(100*attn)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)
    # return '<span style="background-color: {}">{}</span> {}<br>'.format(html_color, word, attn)

def show_bert_explaination(input_ids, attention_mask, attention_weight):
    tokenizer = AutoTokenizer.from_pretrained('./roberta_dic/')

    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_weight = attention_weight.cpu()[0].numpy()
    attention_mask = attention_mask.cpu()[0].numpy()
    # print(attention_weight)

    special_index_list = np.array([i for i, token in enumerate(tokens) if token != "[CLS]" and token != "[SEP]" and token != "[EOS]" and token != "[PAD]" and token != "[DESC]" and token != "。"])

    input_ids = input_ids[special_index_list]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    attention_weight = attention_weight[special_index_list]

    html_outputs = []

    # attention_weight = min_max(attention_weight)

    for word, attn in zip(tokens, attention_weight):
        html_outputs.append(highlight_r(word, attn))

    return html_outputs


def visualize_attention_test(src, tgt, model, idx, device):
    input_ids = src["input_ids"].to(device)
    token_type_ids = src["token_type_ids"].to(device)
    attention_mask = src["attention_mask"].to(device)
    output = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)
    logits = output["logits"].squeeze()
    attentions = output["attentions"]
    html_dic = {}
    preds = torch.tanh(logits)
    attentions = torch.stack(list(attentions)).sum(0) / len(attentions)
    # attentions = attentions[-1].sum(1)[:, 0, :]
    for input_id, score, pred, attention in zip(input_ids, tgt, preds, attentions):
        attention = attention.sum(0)
        html = [f'正解スコア: {score.item()}<br>推定スコア: {pred.item()}<br>']
        html.extend(show_bert_explaination(input_id, attention_mask, attention))
        html.append('<br><br>')
        html_dic[' '.join(html)] = pred.item()
        # for i, head in enumerate(attention):
        #     html = [f'HEAD: {i+1} 正解スコア: {score.item()}<br>推定スコア: {pred.item()}<br>']
        #     html.extend(show_bert_explaination(input_id, attention_mask, head))
        #     html.append('<br><br>')
        #     html_dic[' '.join(html)] = pred.item()

    sorted_html_dic = sorted(html_dic.items(), key=itemgetter(1), reverse=True)
    html = [html for html, _ in sorted_html_dic]

    return preds, html
import json
from copy import deepcopy
from collections import OrderedDict

from pyknp import Juman
jumanpp = Juman(timeout=180)

import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import Dataset

from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split_info_file: str,
        split_id: int,
        data_type: str,
        if_down_scale: bool,
        max_len: int,
        vocab_dir: str
    ):
        with open(split_info_file) as f:
            split_info_jd = json.load(f, object_pairs_hook=OrderedDict)
        filename_list = split_info_jd[split_id][data_type]

        self.data_type = data_type
        self.max_len = max_len

        self.tokenizer = AutoTokenizer.from_pretrained(vocab_dir)
        self.pad_id = self.tokenizer.pad_token_id
        self.desc_id = self.tokenizer.vocab["[DESC]"]

        self.data_list = self.get_data_list(data_dir, filename_list)

        if self.data_type == "train":
            self.data_list = sum(self.data_list, [])

        self.len = len(self.data_list)


    def __len__(self):
        return self.len


    def __getitem__(self, index):
        if self.data_type == "train":
            src = self.data_list[index]["src"]
            tgt = self.data_list[index]["score"]
            return src, tgt
        else:
            data_list = self.data_list[index]
            return data_list


    def score_scaling(self, score, down_scale=False):
        if not down_scale:
            return (score - 3) / 2.0
        if score == 2:
            return self.score_scaling(1)
        elif score == 4:
            return self.score_scaling(5)
        else:
            return self.score_scaling(score)


    # 話者ごとに入れ子になったdictのlistを返す
    def get_data_list(self, data_dir, filename_list):
        data_list = []
        for filename in filename_list:
            with open(data_dir+filename) as f:
                chat_rec_jd = json.load(f, object_pairs_hook=OrderedDict)

            # 発話リスト(BERTの入力の形)取得
            speaker2uttrlist = {}
            for uttr_dict in chat_rec_jd["dialogue"]:
                uttr = uttr_dict["utterance"]
                speaker = uttr_dict["speaker"]
                juman_uttr = [mrph.midasi for mrph in jumanpp.analysis(uttr).mrph_list()]
                encode_dict = self.tokenizer(" ".join(juman_uttr), add_special_tokens=True)
                if speaker in speaker2uttrlist:
                    for k in speaker2uttrlist[speaker].keys():
                        speaker2uttrlist[speaker][k].extend(encode_dict[k][1:])
                else:
                    speaker2uttrlist[speaker] = encode_dict

            for speaker, elem in chat_rec_jd["questionnaire"].items():
                tmp_data_list = []
                for eval_dict in elem["evaluation"]:
                    data_dict = {"filename": filename, "speaker":speaker}
                    spot_id = eval_dict["id"]
                    score = self.score_scaling(eval_dict["score"], False)
                    #print(score)
                    data_dict.update([("id", spot_id), ("score", score)])

                    for place_dict in chat_rec_jd["place"]:
                        if place_dict["id"] == spot_id:
                            description = place_dict["description"]

                    src_dict = deepcopy(speaker2uttrlist[speaker])

                    for k in src_dict.keys():
                        src_length = len(src_dict[k])
                        if src_length >= 256:
                            src_dict[k] = [src_dict[k][0]]+src_dict[k][2:128] + src_dict[k][-128:-1]
                        else:
                            src_dict[k] = src_dict[k][:-1]
                    # DESCトークン
                    src_dict["input_ids"].append(self.desc_id)
                    src_dict["token_type_ids"].append(0)
                    src_dict["attention_mask"].append(1)


                    description = description.replace("\r", "").replace("\n", "")
                    juman_desc = [mrph.midasi for mrph in jumanpp.analysis(description).mrph_list()]

                    # description
                    encode_dict = self.tokenizer(" ".join(juman_desc), add_special_tokens=True)
                    # encode_dict["token_type_ids"] = [1 for _ in range(len(encode_dict["token_type_ids"]))]
                    for k in src_dict.keys():
                        # 戦闘末尾256
                        tokenized_description = encode_dict[k][1:-1] # 先頭の[CLS]と末尾の[SEP]を削除
                        tokenized_description_length = len(tokenized_description)
                        if tokenized_description_length >= 128:
                            if len(src_dict[k]) < 385:
                                src_dict[k].extend(tokenized_description[:128]+tokenized_description[-128:])
                            else:
                                src_dict[k].extend(tokenized_description[:128])
                        else:
                            src_dict[k].extend(tokenized_description)
                        src_length = len(src_dict[k])
                        if src_length < 512:
                            tmp_obj = torch.tensor(src_dict[k][:-1])
                            # padding: input_idsはPAD_ID埋め，token_type_idsは0埋め，attention_maskは[PAD]の箇所0
                            packed_obj = pack_sequence([tmp_obj])
                            padding_id = self.pad_id if k == "input_ids" else 0
                            padded_src = pad_packed_sequence(packed_obj, batch_first=True, padding_value=padding_id, total_length=self.max_len)
                            src_dict[k] = padded_src[0][0]
                        else:
                            src_dict[k] = torch.tensor(deepcopy(src_dict[k]))


                        if len(src_dict[k]) != 512:
                            print("not fit length!", len(src_dict[k]))

                    #print(src_dict["input_ids"])
                    #print(self.tokenizer.decode(src_dict["input_ids"]))
                    data_dict["src"] = src_dict

                    tmp_data_list.append(data_dict)
                data_list.append(tmp_data_list)

        return data_list


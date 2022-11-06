# ChatRec
<!-- 日本語版のREADMはこちら→[日本語版](https://github.com/Ryutaro-A/ChatRec/blob/main/README_JA.md) -->

This is the official implementation of the following paper: Ryutaro Asahara, Masaki Takahashi, Chiho Iwahashi and Michimasa Inaba. ChatRec: A Dataset for Tourist Spot Recommendation using Chat Dialogue. 2022.

>Abstract<br>
>This paper introduces ChatRec, a new dataset of recommendations of tourist spots provided using chat dialogues. ChatRec incorporates 1,005 human dialogues and 15,813 human evaluation scores of tourist spots.  Because target tourist spots are rarely mentioned in the dialogues, the speaker’s interests and the information provided on tourist spots require integration in order to provide appropriate recommendations.<br>
>We present the benchmark performance of ChatRec using several methods of evaluation, including traditional word-based approaches and pretrained neural language models. The results of the experiment suggest that it is challenging to extract appropriate information from dialogues even when pretrained neural language models are used.

## Overview
The code and sample data for our work is organized as:

* `scripts/` contains the main model scripts
* `data/chat_and_rec/` has our dataset
* `data/crossval_split/` has a json file specifying the division method for cross-validation


## Requirements
1. The implementation is based on Python 3.x. To install the dependencies used, run:
```.bash
$ pip install -r requirements.txt
```
2. Save the pretrained word2vec model(jawiki.all_vectors.200d.txt from [here](https://github.com/singletongue/WikiEntVec/releases))
3. Install MeCab and save the mecab-ipadic-NEologd(v0.0.6 from [here](https://github.com/neologd/mecab-ipadic-neologd))
4. Install Juman++ 2.0.0-rc3

## Get Started

Runinng train, test, evaluations scripts if you excecute `run.sh`.

```.bash
sh run.sh
```

When changing the method or data type, change the options according to the table below.

| Args               | Desctiption                                                                                                            |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| data_type          | You can chosse no_restriction or travel or except_for_travel.                                                          |
| method             | You can chosse human or tfidf_cossim or w2v_cossim, or w2v_svr or bert-base or roberta-base or roberta-large.         |
| data_dir           | Please fill in the data directory.                                                                                       |
| split_info_dir     | Enter the location of the directory containing the file that describes how the data was split.                               |
| split_id           | Enter the division ID of the data.                                                                                           |
| model_output_dir   | Enter the location where you would like to save the model.                                                                               |
| output_dir         | Enter the location where you want to store the results of the predictions made by the chosen method.                                                                       |
| vocab_dir          | Enter the directory of the dictionary to be used for the pre-trained Transformer.(`./data/roberta_dic/` or `./data/BERT/`)                                                        |
| mecab_dict         | Enter the specified directory for mecab dictionaries used for tfidf and word2vec.                                               |
| word2vec_file      | Enter the path to your pre-studied word2vec.                                                                             |
| batch_size         | Batch size when training a Transformer model.                                                                            |
| max_epoch          | Epoch size when training a Transformer model.                                                                             |
| patience           | Epoch size until learning is terminated if the minimum loss is not updated.                                           |
| max_len            | Maximum token size including dialog history and tourist attraction descriptions.                                                         |
| optimizer          | Optimization function used to train the Transformer model.                                                            |
| lr                 | Transformer model learning rate.                                                                                |
| hidden_dropout     | Dropout rates for the hidden layer of the Transformer model.                                                              |
| attention_dropout  | Dropout rate for the attention layer of the Transformer model.                                                         |
| use_pretrain_model | With this option, the GPU is used to train the Transformer model. Without this option, only the model structure is used.           |
| use_cuda           | With this option, the GPU is used to train the Transformer model.                                                   |
| use_device_ids     | By specifying this option and a number, you can specify the ID of the GPU device to be used for training. If not specified, all GPUs are used. |

## License
This software is released under the MIT License, see LICENSE.txt.

## Contacts

Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)

Mail: ryu1104.as[at]gmail.com

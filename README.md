# ChatRec
日本語版のREADMはこちら→[日本語版](https://github.com/Ryutaro-A/ChatRec/blob/main/README_JA.md)

## Get Started

The following shell script can be easily executed.

```
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

## Contacts

Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)

Mail: ryu1104.as[at]gmail.com

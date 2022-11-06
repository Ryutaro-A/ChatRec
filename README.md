# ChatRec

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
| data_dir           | データのあるパスを指定します．                                                                                         |
| split_info_dir     | 前処理で作成した，データの分割方法を記載したファイルのあるディレクトリの場所を指定する．                               |
| split_id           | データの分割IDを指定します．                                                                                           |
| model_output_dir   | 学習したモデルの保存場所を指定します．                                                                                 |
| output_dir         | 手法によって予測した結果の保存場所を指定します．                                                                       |
| vocab_dir          | 事前学習済みTransformerに用いる辞書のディレクトリを指定します．                                                        |
| mecab_dict         | tfidfおよびword2vecに用いるmecabの辞書を指定のディレクトリを指定します．                                               |
| word2vec_file      | 事前学習済みのword2vecのパスを指定します．                                                                             |
| batch_size         | Transformerモデルの学習の際のバッチサイズ．                                                                            |
| max_epoch          | Transformerモデルの学習の際のエポック数．                                                                              |
| patience           | 損失の最小値が更新されない場合の学習を打ち切るまでのエポック数を指定します．                                           |
| max_len            | 対話履歴と観光地説明文を含めたトークン数の最大数を指定します．                                                         |
| optimizer          | Transformerモデルの学習に用いる最適化関数を指定します．                                                                |
| lr                 | Transformerモデルの学習率を指定します．                                                                                |
| hidden_dropout     | Transformerモデルの隠れ層のドロップアウト率を指定します．                                                              |
| attention_dropout  | TransformerモデルのAttention層のドロップアウト率を指定します．                                                         |
| use_pretrain_model | このオプションを付けることでTransformerモデルの学習にGPUを用います．付けない場合はモデル構造のみを用います．           |
| use_cuda           | このオプションを付けることでTransformerモデルの学習にGPUを用います．                                                   |
| use_device_ids     | このオプションと数字を指定することで，学習に用いるGPUデバイスのIDを指定できます．指定しない場合は全GPUが使用されます． |

## License
This software is released under the MIT License, see LICENSE.txt.

## Contacts

Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)

Mail: ryu1104.as[at]gmail.com

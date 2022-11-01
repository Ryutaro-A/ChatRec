# ChatRec

## Get Started

交差検証を行うためのデータ分割を行います．

```bash
python scripts/util/crosval_split.py
```

以下のシェルスクリプトを実行することで簡単に実行ができます．

```
sh run.sh
```

手法やデータ・タイプを変更する際は以下の表に従ってオプションを変更してください．

| オプション         | 例                                 | 説明文                                                                                                                 | 引数                                                                             |
| ------------------ | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| data_type          | travel                             | データの種類を指定します．                                                                                             | no_restriction, travel, except_for_travel                                        |
| method             | roberta-base                       | 用いる手法を指定します．                                                                                               | human, tfidf_cossim, w2v_cossim, w2v_svr, bert-base, roberta-base, roberta-large |
| data_dir           | ./data/chatrec/                    | データのあるパスを指定します．                                                                                         | -                                                                                |
| split_info_dir     | ./data/crossval_split/             | 前処理で作成した，データの分割方法を記載したファイルのあるディレクトリの場所を指定する．                               | -                                                                                |
| split_id           | 2                                  | データの分割IDを指定します．                                                                                           | -                                                                                |
| model_output_dir   | ./save_model/                      | 学習したモデルの保存場所を指定します．                                                                                 | -                                                                                |
| output_dir         | ./result/                          | 手法によって予測した結果の保存場所を指定します．                                                                       | -                                                                                |
| vocab_dir          | ./data/roberta_dic/                | 事前学習済みTransformerに用いる辞書のディレクトリを指定します．                                                        | -                                                                                |
| mecab_dict         | ./data/mecab/mecab_dic/            | tfidfおよびword2vecに用いるmecabの辞書を指定のディレクトリを指定します．                                               | -                                                                                |
| word2vec_file      | ./data/jawiki.all_vectors.200d.txt | 事前学習済みのword2vecのパスを指定します．                                                                             | -                                                                                |
| batch_size         | 128                                | Transformerモデルの学習の際のバッチサイズ．                                                                            |                                                                                  |
| max_epoch          | 100                                | Transformerモデルの学習の際のエポック数．                                                                              |                                                                                  |
| patience           | 20                                 | 損失の最小値が更新されない場合の学習を打ち切るまでのエポック数を指定します．                                           |                                                                                  |
| max_len            | 512                                | 対話履歴と観光地説明文を含めたトークン数の最大数を指定します．                                                         |                                                                                  |
| optimizer          | Adafactor                          | Transformerモデルの学習に用いる最適化関数を指定します．                                                                | Adafactor, Adam, AdamW, SGD                                                      |
| lr                 | 0.0001                             | Transformerモデルの学習率を指定します．                                                                                |                                                                                  |
| hidden_dropout     | 0.1                                | Transformerモデルの隠れ層のドロップアウト率を指定します．                                                              |                                                                                  |
| attention_dropout  | 0.1                                | TransformerモデルのAttention層のドロップアウト率を指定します．                                                         |                                                                                  |
| use_pretrain_model | -                                  | このオプションを付けることでTransformerモデルの学習にGPUを用います．付けない場合はモデル構造のみを用います．           |                                                                                  |
| use_cuda           | -                                  | このオプションを付けることでTransformerモデルの学習にGPUを用います．                                                   |                                                                                  |
| use_device_ids     | 0123                               | このオプションと数字を指定することで，学習に用いるGPUデバイスのIDを指定できます．指定しない場合は全GPUが使用されます． |                                                                                  |


## Contacts
Twitter: [@ryu1104_m](https://twitter.com/ryu1104_m)

Mail: ryu1104.as[at]gmail.com
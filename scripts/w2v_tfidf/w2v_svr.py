from sklearn.feature_extraction.text import TfidfVectorizer
import random
import json
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from sklearn.svm import SVR
import pickle
# w2v = KeyedVectors.load_word2vec_format("data/jawiki.all_vectors.200d.txt", binary=False)


# tagger = MeCab.Tagger("-Owakati -d /export/share/lib/mecab/dic/mecab-ipadic-neologd-0.0.6/")
# tagger.parse("")

def calc_idf_weighted_vector(text, word2idf, w2v):
    out_vec = None
    count = 0
    for w in text.split():
        if w in w2v and w in word2idf:
            if out_vec is None:
                out_vec = w2v[w] * word2idf[w]
            else:
                out_vec += w2v[w] * word2idf[w]
            count += 1
    out_vec = out_vec / count
    return out_vec


def score2y(score):
    if 1 == score:
        return -1.0
    elif 2 == score:
        return -0.5
    elif 3  == score:
        return 0.0
    elif 4  == score:
        return 0.5
    elif 5 == score:
        return 1.0
    else:
        print("スコアエラー")
        return 0.0


def extract_document(filename, tagger):
    with open(filename) as f:
        jd = json.load(f)

    # 話者名を取得
    spk1 = list(jd["questionnaire"].keys())[0]
    spk2 = list(jd["questionnaire"].keys())[1]

    spk1_text = ""
    spk2_text = ""
    # 対話を話者ごとに連結
    for d in jd["dialogue"]:
        if spk1 == d["speaker"]:
            spk1_text += tagger.parse(d["utterance"]).strip() + " "
        elif spk2 == d["speaker"]:
            spk2_text += tagger.parse(d["utterance"]).strip() + " "
        else:
            print("スピーカーが存在しません", filename, d)


    # 説明文を取得
    desc = []
    for p in jd["place"]:
        desc.append(tagger.parse(p["description"]).strip())

    doc = desc + [spk1_text] + [spk2_text]

    # スコアを取得
    spk1_score = []
    for e in jd["questionnaire"][spk1]["evaluation"]: 
        spk1_score.append(score2y(e["score"]))
    spk2_score = []
    for e in jd["questionnaire"][spk2]["evaluation"]: 
        spk2_score.append(score2y(e["score"]))

    return spk1, spk2, spk1_text, spk2_text, spk1_score, spk2_score, desc, doc


def run(args):
    if args.mecab_dict is None:
        tagger = MeCab.Tagger("-Owakati")
    else:
        tagger = MeCab.Tagger("-Owakati -d " + args.mecab_dict)
    tagger.parse("")

    print("Word2vec File Read Start...")
    w2v = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=False)
    print("Finishd")

    split_info_file = f'{args.split_info_dir}{args.data_type}_split.json'
    with open(split_info_file) as f:
        split = json.load(f)[args.split_id]
    train_files = split["train"] + split["dev"]
    test_files = split["test"]

    # TfidfVectorizerのfit
    doc = []
    for tf in train_files:
        _, _, _, _, _, _, _, t = extract_document(args.data_dir+tf, tagger)
        doc += t

    vectorizer = TfidfVectorizer(tokenizer=lambda x:x.split())
    vectorizer.fit(doc)

    word2idf = {}
    vocabs = vectorizer.get_feature_names()
    idfs = vectorizer.idf_
    for v, idf in zip(vocabs, idfs):
        word2idf[v] = idf

    # SVRの学習
    x = []
    y = []
    for tf in train_files:
        spk1, spk2, spk1_text, spk2_text, spk1_score, spk2_score, desc, doc  = extract_document(args.data_dir + tf, tagger)
        sp1v = calc_idf_weighted_vector(spk1_text, word2idf, w2v).tolist()
        sp2v = calc_idf_weighted_vector(spk2_text, word2idf, w2v).tolist()
        for di, s1s, s2s in zip(desc, spk1_score, spk2_score):
            dv =calc_idf_weighted_vector(di, word2idf, w2v).tolist()
            x.append(sp1v + dv)
            y.append(s1s)
            x.append(sp2v + dv)
            y.append(s2s)

    print("Training Start")
    model = SVR()
    model.fit(x,y)
    print("Finished")

    if args.model_output_dir is not None:
        os.makedirs(args.model_output_dir, exist_ok=True)
        pickle.dump(model, open(args.model_output_dir + ".bin" , 'wb'))
        print("Model saved: " + args.model_output_dir + ".bin" )


    for tf in test_files:
        spk1, spk2, spk1_text, spk2_text, spk1_score, spk2_score, desc, doc = extract_document(args.data_dir + tf, tagger)
        sp1v = calc_idf_weighted_vector(spk1_text, word2idf, w2v).tolist()
        sp2v = calc_idf_weighted_vector(spk2_text, word2idf, w2v).tolist()

        # 出力用
        out_data = {spk1:[], spk2:[]}
        for idx, di in enumerate(desc):
            dv =calc_idf_weighted_vector(di, word2idf, w2v).tolist()

            out_data[spk1].append({"id": str(idx+1), "score": model.predict([sp1v + dv])[0]})
            out_data[spk2].append({"id": str(idx+1), "score": model.predict([sp2v + dv])[0]})

        # os.makedirs(args.output_dir, exist_ok=True)
        # with open(args.output_dir + tf.replace(".json", ".rmd.json"), "w") as w:
        #     json.dump(out_data, w, ensure_ascii=False, indent=4)

        tmp_dir = tf.split('/')[:-1]
        os.makedirs(args.output_dir+"/".join(tmp_dir), exist_ok=True)
        with open(args.output_dir + tf.replace(".json", ".rmd.json"), "w") as w:
            json.dump(out_data, w, ensure_ascii=False, indent=4)

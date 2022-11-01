from sklearn.feature_extraction.text import TfidfVectorizer
import random
import json
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
import os
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

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
            print("スピーカーが存在しません", file, d)


    # 説明文を取得
    desc = []
    for p in jd["place"]:
        desc.append(tagger.parse(p["description"]).strip())

    doc = desc + [spk1_text] + [spk2_text]
    return spk1, spk2, spk1_text, spk2_text, desc, doc

def run(args, split_id):
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
        split = json.load(f)

    s = split[split_id]
    train_files = s["train"] + s["dev"]
    test_files = s["test"]

    # TfidfVectorizerのfit
    doc = []
    for tf in train_files:
        _, _, _, _, _, t = extract_document(args.data_dir + tf, tagger)
        doc += t

    vectorizer = TfidfVectorizer(tokenizer=lambda x:x.split())
    vectorizer.fit(doc)

    word2idf = {}
    vocabs = vectorizer.get_feature_names()
    idfs = vectorizer.idf_
    for v, idf in zip(vocabs, idfs):
        word2idf[v] = idf


    #コサイン類似度
    for tf in test_files:

        spk1, spk2, spk1_text, spk2_text, desc, _ = extract_document(args.data_dir + tf, tagger)
        sp1v = calc_idf_weighted_vector(spk1_text, word2idf, w2v)
        sp2v = calc_idf_weighted_vector(spk2_text, word2idf, w2v)

        # 出力用
        out_data = {spk1:[], spk2:[]}

        for idx, d in enumerate(desc):
            dv = calc_idf_weighted_vector(d, word2idf, w2v)
            sim1 = float(cosine_similarity([sp1v], [dv])[0][0])
            sim2 = float(cosine_similarity([sp2v], [dv])[0][0])

            out_data[spk1].append({"id": str(idx+1), "score": sim1})
            out_data[spk2].append({"id": str(idx+1), "score": sim2})


        tmp_dir = tf.split('/')[-2]
        os.makedirs(args.output_dir+str(split_id)+'/'+tmp_dir+'/', exist_ok=True)
        with open(args.output_dir + str(split_id) + '/' + tf.replace(".json", ".rmd.json"), "w") as w:
            json.dump(out_data, w, ensure_ascii=False, indent=4)

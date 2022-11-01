from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import random
import json
import MeCab
from sklearn.metrics.pairwise import cosine_similarity
import os


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

    #コサイン類似度
    for tf in test_files:
        spk1, spk2, spk1_text, spk2_text, desc, _ = extract_document(args.data_dir + tf, tagger)
        sp1v = vectorizer.transform([spk1_text])
        sp2v = vectorizer.transform([spk2_text])

        # 出力用
        out_data = {spk1:[], spk2:[]}

        for idx, d in enumerate(desc):
            dv = vectorizer.transform([d])
            out_data[spk1].append({"id": str(idx+1), "score": cosine_similarity(sp1v, dv)[0][0]})
            out_data[spk2].append({"id": str(idx+1), "score": cosine_similarity(sp2v, dv)[0][0]})

        tmp_dir = tf.split('/')[-2]
        os.makedirs(args.output_dir+str(split_id)+'/'+tmp_dir+'/', exist_ok=True)
        with open(args.output_dir + str(split_id) + '/' + tf.replace(".json", ".rmd.json"), "w") as w:
            json.dump(out_data, w, ensure_ascii=False, indent=4)


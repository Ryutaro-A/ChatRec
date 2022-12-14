import numpy as np
import sys
import glob
import json
import os

def dcg_at_k(r, k, method=0):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum((np.power(2, r)-1) / np.log2(np.arange(1, r.size + 1)+1))
    return 0.


def calc_ndcg_at_k(r, k, method=0):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def ndcg_dict(pred, ans, max_k=None):
    if max_k is None:
        max_k = len(pred)
    if pred is not np:
        x = np.array(pred)
    if ans is not np:
        y = np.array(ans)


    sorted_index = np.argsort(-x)
    rs = y[sorted_index].tolist()

    ndcgs = {}
    for i in range(1, max_k + 1):
        ndcgs[i] = 0.0

    for i in range(1, max_k + 1):
        ndcgs[i] = calc_ndcg_at_k(rs, i)
    return ndcgs

def ndcg_at_k(pred_dir, ans_dir):
    if not pred_dir.endswith("/"):
        pred_dir += "/"
    if not ans_dir.endswith("/"):
        ans_dir += "/"

    total_ndcg_dict = {}
    count = 0
    for filename in glob.glob(ans_dir + "*.json"):
        t_scores = {}
        if not os.path.exists(filename.replace(ans_dir, pred_dir).replace(".json", ".rmd.json")):
            continue
        with open(filename) as f1, open(filename.replace(ans_dir, pred_dir).replace(".json", ".rmd.json")) as f2:
            ans_jd = json.load(f1)
            pred_jd = json.load(f2)

            t_scores = {}
            for k, v in ans_jd["questionnaire"].items():
                t_scores[k] = []
                for d in v["evaluation"]:
                    t_scores[k].append(d["score"])
            y_scores = {}
            for k, v in pred_jd.items():
                y_scores[k] = []
                for d in v:
                    y_scores[k].append(d["score"])

            for k in ans_jd["questionnaire"].keys():
                tmp_dict = ndcg_dict(y_scores[k], t_scores[k], 10)

                for k, v in tmp_dict.items():
                    if k in total_ndcg_dict:
                        total_ndcg_dict[k] += v
                    else:
                        total_ndcg_dict[k] = v
                count += 1
    for k in total_ndcg_dict.keys():
        total_ndcg_dict[k] = total_ndcg_dict[k] / count

    return total_ndcg_dict, count

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("?????????????????????????????????????????????????????????????????????python ndcg.py [??????????????????????????????????????????] [?????????????????????????????????????????????]")
        exit()
    pred_dir = sys.argv[1]
    ans_dir = sys.argv[2]

    total_ndcg_dict, count = ndcg_at_k(pred_dir, ans_dir)

    print("NDCG@K", total_ndcg_dict, "DATA_COUNT:", count)




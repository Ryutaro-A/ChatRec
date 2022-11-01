import glob
import json
import os
import sys
from scipy.stats import rankdata
from scipy.stats import spearmanr


args = sys.argv

pred_dir = args[1]
ans_dir = args[2]

files = glob.glob(f'{pred_dir}*.json')
sum_crr = 0
count = 0
for file_path in files:
    with open(file_path, encoding='utf-8') as f:
        pred_json_data = json.load(f)

    filename = os.path.basename(file_path).replace(".rmd", "")

    with open(f'{ans_dir}{filename}', encoding='utf-8') as f:
        ans_json_data = json.load(f)

    speakers = ans_json_data["questionnaire"].keys()
    for speaker in speakers:
        pred_score_list = [data["score"] for data in pred_json_data[speaker]]
        ans_json_list = [data["score"] for data in ans_json_data["questionnaire"][speaker]["evaluation"]]

        # 同順は同じ順位にしてランキングを作成
        pred_score_list = rankdata(pred_score_list)
        ans_score_list = rankdata(ans_json_list)

        # スピアマン相関係数を計算
        correlation, pvalue = spearmanr(pred_score_list, ans_score_list)

        sum_crr += correlation
        count += 1
print(f"Spearman's rank correlation coefficient: {sum_crr/count}")
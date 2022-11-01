import glob
import json
import random
import os

def run(args):
    files = glob.glob(f'{args.data_dir}{args.data_type}/*.json')
    os.makedirs(f'{args.output_dir}/{args.data_type}/', exist_ok=True)
    for file_path in files:
        with open(file_path, mode='r', encoding='utf-8') as f:
            json_data = json.load(f)
        q = json_data["questionnaire"]
        users = list(q.keys())
        user1_est = q[users[0]]["estimation"]
        user2_est = q[users[1]]["estimation"]
        users_est_dic = {users[0]:user2_est, users[1]:user1_est}
        with open(args.output_dir+args.data_type+'/'+os.path.basename(file_path).replace(".json", ".rmd.json"), mode='w', encoding='utf-8') as f:
            json.dump(users_est_dic, f, ensure_ascii=False, indent=4)
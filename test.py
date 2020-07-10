import torch
import torch.nn.functional as F
import os, sys
import json
import urllib.request
import csv
import re
import nltk
import numpy as np

sys.path.insert(0, os.path.dirname(os.getcwd()))
from src.word2vec import word2vec
from src.model import build_bilstm_ram
from src import vrm_config

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

client_id = "" ## enter your id 
client_secret = "" ## enter your key

pos_map = {'LS': 0, 'TO': 1, 'VBN': 2, "''": 3, 'WP': 4, 'UH': 5, 'VBG': 6, 'JJ': 7, 'VBZ': 8, '--': 9,
    'VBP': 10, 'NN': 11, 'DT': 12, 'PRP': 13, ':': 14, 'WP$': 15, 'NNPS': 16, 'PRP$': 17, 'WDT': 18,
    '(': 19, ')': 20, '.': 21, ',': 22, '``': 23, '$': 24, 'RB': 25, 'RBR': 26, 'RBS': 27, 'VBD': 28,
    'IN': 29, 'FW': 30, 'RP': 31, 'JJR': 32, 'JJS': 33, 'PDT': 34, 'MD': 35, 'VB': 36, 'WRB': 37,
    'NNP': 38, 'EX': 39, 'NNS': 40, 'SYM': 41, 'CC': 42, 'CD': 43, 'POS': 44}

def kor2eng(kor):
    encText = urllib.parse.quote(kor)
    data = "source=ko&target=en&text=" + encText
    url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation" # "https://openapi.naver.com/v1/papago/n2mt"
    request = urllib.request.Request(url)
    request.add_header("X-NCP-APIGW-API-KEY-ID", client_id)
    request.add_header("X-NCP-APIGW-API-KEY", client_secret)
    
    # todo: error 500 handling 
    try:
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    except Exception:
        print('retry')
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    #response = urllib.request.urlopen(request, data=data.encode("utf-8"))
    rescode = response.getcode()

    if rescode != 200:
        return None

    else:
        # httpresponse
        response_body = response.read().decode('utf-8')
        res = json.loads(response_body)
        return res["message"]['result']['translatedText']


def preprocess(dialogs):
    previous_caller = None
    utterance_chunk = []
    pos_chunk = []
    mask_chunk = []
    qtag_chunk = []
    caller_chunk = []

    tag_chunk = []

    for message_caller in dialogs:
        tag = message_caller['tag']
        utterance = message_caller['message'].strip()
        utterance = re.sub('[()/\\\{}-]',' ',utterance)

        qtag = 1 if len(utterance) != 0 and utterance[-1] == '?' else 0

        tokens = nltk.word_tokenize(utterance)
        tokens = nltk.pos_tag(tokens)
        words = [token[0] for token in tokens if len(token) == 2]
        true_length = len(words)
        words = _words_to_vector(words)
        poses = [token[1] for token in tokens if len(token) == 2]
        poses =_poses_to_vector(poses)

        if previous_caller and message_caller['caller'] == previous_caller:
            caller = 1
        else:
            caller = 0
        previous_caller = message_caller['caller']

        mask = [ 1 for _ in range(true_length)][:sent_len] + [0 for _ in range(sent_len-true_length)]

        utterance_chunk.append(words)
        pos_chunk.append(poses)
        mask_chunk.append(mask)
        qtag_chunk.append(qtag)
        caller_chunk.append(caller)

        tag_chunk.append(tag)

    utterance_chunk = torch.stack(utterance_chunk)
    pos_chunk = torch.tensor(pos_chunk).to(utterance_chunk.dtype)
    mask_chunk = torch.tensor(mask_chunk).to(utterance_chunk.dtype)
    qtag_chunk = torch.tensor(qtag_chunk).to(utterance_chunk.dtype)
    caller_chunk = torch.tensor(caller_chunk).to(utterance_chunk.dtype)

    return [utterance_chunk, pos_chunk,mask_chunk, qtag_chunk, caller_chunk], tag_chunk

def _poses_to_vector(poses):
    pos_index = [pos_map[p] for p in poses]

    pos_vector = np.zeros((sent_len,pos_tag_len))
    pos_index = pos_index[:sent_len]

    for i,j in enumerate(pos_index):
        pos_vector[i][j]=1

    return pos_vector

def classify_vrm(dicts):
    keys = dicts.keys()

    results = {}
    for key in keys:
        data, tag = preprocess(dicts[key])
        data = [d.to(device).unsqueeze(0) for d in data]
        preds = model_vrm(data)
        results[key] = {}
        results[key]['is_correct_f'] = []
        results[key]['is_correct_i'] = []
        results[key]['ans_f'] = []
        results[key]['ans_i'] = []

        for idx, pred in enumerate(preds):
            pred = pred.squeeze(0)
            # 'form_SE', 'form_PE', 'form_FR', 'intent_SE', 'intent_PE', 'intent_FR'
            x = F.softmax(pred[0:2], -1)[-1] > 0.5 # 1: speaker 0: other
            y = F.softmax(pred[2:4], -1)[-1] > 0.5
            z = F.softmax(pred[4:6], -1)[-1] > 0.5

            if x and y and z:
                f = 'D'
            elif x and y and not z:
                f = 'E'
            elif x and not y and z:
                f = 'A'
            elif x and not y and not z:
                f = 'C'
            elif not x and y and z:
                f = 'Q'
            elif not x and y and not z:
                f = 'K'
            elif not x and not y and z:
                f = 'I'
            elif not x and not y and not z:
                f = 'R'

            x = F.softmax(pred[6:8], -1)[-1] > 0.5
            y = F.softmax(pred[8:10], -1)[-1] > 0.5
            z = F.softmax(pred[10:12], -1)[-1] > 0.5

            if x and y and z:
                i = 'D'
            elif x and y and not z:
                i = 'E'
            elif x and not y and z:
                i = 'A'
            elif x and not y and not z:
                i = 'C'
            elif not x and y and z:
                i = 'Q'
            elif not x and y and not z:
                i = 'K'
            elif not x and not y and z:
                i = 'I'
            elif not x and not y and not z:
                i = 'R'

            if tag[idx][0] in ['D', 'E', 'A', 'C', 'Q', 'K', 'I', 'R']:
                #results[key]['ans_f'].append(f)
                results[key]['is_correct_f'].append(int(f == tag[idx][0]))

            if tag[idx][1] in ['D', 'E', 'A', 'C', 'Q', 'K', 'I', 'R']:
                #results[key]['ans_i'].append(i)
                results[key]['is_correct_i'].append(int(i == tag[idx][1]))
            results[key]['ans_f'].append(f)
            results[key]['ans_i'].append(i)

    return results


# load model
parent_dir = os.path.dirname(os.getcwd())
embedding_path = os.path.join(parent_dir,'VRM_recognition/data','word2vec_from_glove.bin')
#embedding_path = os.path.join(parent_dir,'AIF-2019/data','word2vec_from_glove.bin')
sent_len = vrm_config.sent_len

pos_tag_len = 45
print("getting embedding...")
wv = word2vec(embedding_path, sent_len=sent_len)
_words_to_vector = wv.to_vector
print("done..!")

print("setting models...")

use_cuda = True
device = torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu")

model_vrm = build_bilstm_ram(device, vrm_config)
model_vrm = model_vrm.to(device)
model_vrm.load_state_dict(torch.load(os.path.join(parent_dir,'VRM_recognition/trained_model','trained_model_VRM.pth')))
model_vrm.eval()

print("done...!")

# read excel file
data = {}

#print(kor2eng('안녕하세요'))

#f = open('kor-eng.csv', 'w', encoding='utf-8', newline='')
#wr = csv.writer(f)

with open('relabeling.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    cur_key = None
    for row in csv_reader:
        if line_count > 0:
            # row[0] = conversation_id, row[1] = utterance_id, row[2] = utterance, row[3] = mode, row[4] = caller
            if cur_key is None or cur_key != row[0]: # new dialogue
                data[row[0]] = []
                cur_key = row[0]
            tag = row[3].split('/')[0]
            #ans = kor2eng(row[-2])
            #wr.writerow([line_count, row[-2], ans])
            data[row[0]].append({'message': kor2eng(row[2]), 'caller': row[4], 'tag': tag})
        line_count += 1
        print('line %d was processed' % (line_count-1))
    print('Processed %d lines.' % (line_count-1))

res = classify_vrm(data)
key = res.keys()

is_correct_f = []
is_correct_i = []
f_ans = []
i_ans = []

for key in res.keys():
    is_correct_f.extend(res[key]['is_correct_f'])
    is_correct_i.extend(res[key]['is_correct_i'])
    f_ans.extend(res[key]['ans_f'])
    i_ans.extend(res[key]['ans_i'])

print('Form acc: %.3f, Intent acc: %.3f' % (np.mean(is_correct_f), np.mean(is_correct_i)))

#f.close()

import csv

g = open('./relabeling.csv', 'r', encoding='utf-8')
rdr = csv.reader(g)

f = open('output.csv', 'w', encoding='utf-8')
wr = csv.writer(f)
i = 0
for line in rdr:
    if i == 0:
        wr.writerow(['conversation_id','groundtruth','caller', 'cls_output'])#,'caller','kor','eng'])
    else:
        cid, uid, utt, mode, caller = line
        ans = ''.join([f_ans[i-1], i_ans[i-1]])
        wr.writerow([cid, mode, caller, ans])
    i += 1

f.close()
g.close()

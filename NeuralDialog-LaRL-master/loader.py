import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy


def process(input, output, dummy = True):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    for line in tqdm(f):
        lines=json.loads(line.strip())
        seekerid=lines["initiatorWorkerId"]
        recommenderid=lines["respondentWorkerId"]
        contexts=lines['messages']
        movies=lines['movieMentions']
        altitude=lines['respondentQuestions']
        initial_altitude=lines['initiatorQuestions']
        l = ""
        if (altitude and initial_altitude):
            if(dummy):
                l += "<input> 1 1 1 1 1 1 </input> " 
            # else:
            #     l += "<input> "
            #     for key, a in initial_altitude.items():
            #         l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            #     l += "</input> "
            l += "<dialogue> "
            last = ""
            for m in contexts:
                if m['senderWorkerId'] == seekerid:
                    l += "YOU: "
                    last = "THEM: <selection> "
                elif m['senderWorkerId'] == recommenderid:
                    l += "THEM: "
                    last = "YOU: <selection> "
                else:
                    pass
                l += m['text']
                l += " <eos> "
            l += last
            l += "</dialogue> " 
            if(dummy):
                l += "<output> item0=1 item1=1 item2=1 item0=1 item1=1 item2=1 </output> <partner_input> 1 1 1 1 1 1 </partner_input>"
            # else:
            #     l += "<output> "
            #     count1 = 0
            #     for a in initial_altitude.values():
            #         l += ("item%d=" % (count1))
            #         l += str(a['suggested']) + " "
            #         count1 += 1
            #     count2 = 0
            #     for a in altitude.values():
            #         l += ("item%d=" % (count2))
            #         l += str(a['liked']) + " "
            #         count2 += 1
            #     l += "</output> "
            #     l += "<partner_input> "
            #     for key, a in altitude.items():
            #         l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            #     l += "</partner_input> "
            l += " <user> "
            for key, a in altitude.items():
                l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            l += "</user> "
            l = l.replace('\r', '')
            l = l.replace('\n', '')
            l += "\n"
            print(l)
            f2.write(l)

def movie(input, output, dummy = True):
    f = open(input, encoding='utf-8')
    f2 = open(output, "a", encoding='utf-8')
    for line in tqdm(f):
        lines=json.loads(line.strip())
        initial_altitude=lines['initiatorQuestions']
        if initial_altitude: 
            l = "" 
            for key, a in initial_altitude.items():
                l += ("%s %s %s %s " % (key, a['suggested'], a['seen'], a['liked']))
            l += "\n"
            print(l)
            f2.write(l)


if __name__=='__main__':
    process('data/raw/train_data.jsonl', 'data/negotiate/train.txt')
    process('data/raw/valid_data.jsonl', 'data/negotiate/val.txt')
    process('data/raw/test_data.jsonl', 'data/negotiate/test.txt')
    # movie('train_data.jsonl', 'train_movie.txt')
    # movie('valid_data.jsonl', 'val_movie.txt')
    # process('train_data.jsonl', 'train.txt', False)
    # process('valid_data.jsonl', 'val.txt', False)
    # process('test_data.jsonl', 'test.txt', False)
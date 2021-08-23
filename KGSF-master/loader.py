import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy

class dataset(object):
    def __init__(self,filename,opt):
        self.entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
        self.entity_max=len(self.entity2entityId)

        self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))
        self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']
        #self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        f=open(filename,encoding='utf-8')
        self.data=[]
        self.corpus=[]
        for line in tqdm(f):
            lines=json.loads(line.strip())
            seekerid=lines["initiatorWorkerId"]
            recommenderid=lines["respondentWorkerId"]
            contexts=lines['messages']
            movies=lines['movieMentions']
            altitude=lines['respondentQuestions']
            initial_altitude=lines['initiatorQuestions']
            cases=self._context_reformulate(contexts,movies,altitude,initial_altitude,seekerid,recommenderid)
            self.data.extend(cases)
            print(lines)

        #if 'train' in filename:

        #self.prepare_word2vec()

        # self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])

        #self.co_occurance_ext(self.data)
        #exit()

if __name__=='__main__':
    ds=dataset('data/train_data.jsonl')
    print()

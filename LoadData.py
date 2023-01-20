import torch
import gensim.downloader as api
import json
from gensim.models.word2vec import KeyedVectors
from multiprocessing import Pool
import numpy as np
from nltk import SnowballStemmer

model_to_size = {
    'glove-wiki-gigaword-300':300,
    'word2vec-google-news-300':300,
    'glove.6B.300d.txt':300
}

MODEL = 'glove.6B.300d.txt'
EMBEDDINGS_SIZE = model_to_size[MODEL]
FROM_FILE = True

class word_emb():
    def __init__(self) -> None:
        self.embeddings = KeyedVectors(EMBEDDINGS_SIZE)
        self.indexToVec = None
        self.wordToNum = {}

emb = word_emb()

label_to_num = {
    'neutral': 0,
    'entailment': 1,
    'contradiction': 2
}

class enteilmentDataSet(torch.utils.data.Dataset):
    def __init__(self, data_lines:list, num_classes) -> None:
        super().__init__()
        stem = SnowballStemmer('english')
        self.data = []
        self.labels = []
        self.num_classes = num_classes
        for line in data_lines:
            record = json.loads(line)
            label = label_to_num.get(record["gold_label"],3)
            if label!=3:
                sent1 = stem.stem(record["sentence1"][:-1])
                sent2 = stem.stem(record["sentence2"][:-1])
                self.data.append([sent1,sent2])
                for word in sent1.split():
                    try:
                        emb.wordToNum[word] = emb.wordToNum.get(word,emb.embeddings.key_to_index[word]+1)
                    except KeyError:
                        pass
                for word in sent2.split():
                    try:
                        emb.wordToNum[word] = emb.wordToNum.get(word,emb.embeddings.key_to_index[word]+1)
                    except KeyError:
                        pass
                self.labels.append(label if label<self.num_classes else label == 1)
        self.end = len(self.data)
        self.start=0

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        (prem, hypp), y = self.data[idx], self.labels[idx]
        prem_num = []
        for idx,word in enumerate(prem.split()):
            prem_num.append(emb.wordToNum.get(word,0))
        hypp_num = []
        for idx,word in enumerate(hypp.split()):
            hypp_num.append(emb.wordToNum.get(word,0))
        pair = (prem_num , hypp_num)
        return pair, y
    
def collect_batch(batch):
    max_sent = max([max(len(item[0][0]),len(item[0][1])) for item in batch])
    return torch.tensor([
        item[0][0] + [0]* (max_sent-len(item[0][0])) + 
        item[0][1] + [0]* (max_sent-len(item[0][1])) 
        for item in batch],dtype=torch.long), torch.tensor([item[1] for item in batch],dtype=torch.long)

def get_pair(line):
    words = line.split(' ')
    key = words[0]
    vec = np.array(words[1:]).astype(np.float32)
    return key,vec

        
def load_from_file(model, num_of_threads):
    path = 'glove.6B/' + model
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        count = len(lines)
        res = KeyedVectors(EMBEDDINGS_SIZE,count=count)
        with Pool(num_of_threads) as p:
            lines = p.map(get_pair,lines,chunksize=32)
            for key,vec in lines:
                res.add_vector(key=key,vector=vec)
        
        return res


def get_embeddings(from_file, num_of_threads):
    zero = torch.zeros(EMBEDDINGS_SIZE)
    emb.embeddings = load_from_file(MODEL, num_of_threads) if from_file else api.load(MODEL)
    emb.indexToVec = torch.cat(
        [zero[None],
        torch.tensor(
            emb.embeddings.vectors
            )]
        ,dim=0)

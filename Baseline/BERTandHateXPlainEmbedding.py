import transformers
from transformers import AutoTokenizer
from transformers import BertForTokenClassification, BertForSequenceClassification,BertPreTrainedModel, BertModel
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import *
from transformers import BertTokenizer, BertModel


FOLDER_NAME = 'Embeddings/'

import pickle

with open('Data/train_transcriptions.pkl','rb') as fp:
    train_transCript = pickle.load(fp)

with open('Data/train_list.pkl','rb') as fp:
    train_list = pickle.load(fp)

with open('Data/val_transcriptions.pkl','rb') as fp:
    val_transCript = pickle.load(fp)

with open('Data//val_list.pkl','rb') as fp:
    val_list = pickle.load(fp)
    
with open('Data/test_transcriptions.pkl','rb') as fp:
    test_transCript = pickle.load(fp)

with open('Data/test_list.pkl','rb') as fp:
    test_list = pickle.load(fp)
  


model = Model_Rational_Label.from_pretrained("bert-base-uncased-hatexplain-rationale-two")


class Text_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=Model_Rational_Label.from_pretrained("bert-base-uncased-hatexplain-rationale-two", output_hidden_states = True)
        
    def forward(self,x,mask):
        embeddings = self.model(x, mask)
        return embeddings


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased-hatexplain-rationale-two")
def tokenize(sentences, padding = True, max_len = 512):
    input_ids, attention_masks, token_type_ids = [], [], []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(sent,
                                                    add_special_tokens=True,
                                                    max_length=max_len, 
                                                    padding='max_length', 
                                                    return_attention_mask = True,
                                                    return_tensors = 'pt', 
                                                    truncation = True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    return {'input_ids': input_ids, 'attention_masks': attention_masks}


model2 = Text_Model()

allEmbedding ={}
for i in tqdm(train_list):
  try:
    print(i)
    apr = tokenize([train_transCript[i]])
    with torch.no_grad():
        allEmbedding[i]= (model2(apr['input_ids'], apr['attention_masks'])[2][0]).detach().numpy()
    del(apr)
  except:
    pass

for i in tqdm(val_list):
  try:
    print(i)
    apr = tokenize([val_transCript[i]])
    with torch.no_grad():
        allEmbedding[i]= (model2(apr['input_ids'], apr['attention_masks'])[2][0]).detach().numpy()
    del(apr)
  except:
    pass

for i in tqdm(test_list):
  try:
    print(i)
    apr = tokenize([test_transCript[i]])
    with torch.no_grad():
        allEmbedding[i]= (model2(apr['input_ids'], apr['attention_masks'])[2][0]).detach().numpy()
    del(apr)
  except:
    pass


len(allEmbedding)
with open(FOLDER_NAME+'all_HateXPlainembedding.p', 'wb') as fp:
    pickle.dump(allEmbedding,fp)



tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")


allEmbedding ={}
for i in tqdm(train_list):
  try:
    inputs = tokenizer(train_transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding[i]= last_hidden_states[0][0].detach().numpy()
    del(outputs)
  except:
    pass

for i in tqdm(val_list):
  try:
    inputs = tokenizer(val_transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding[i]= last_hidden_states[0][0].detach().numpy()
    del(outputs)
  except:
    pass

for i in tqdm(test_list):
  try:
    inputs = tokenizer(test_transCript[i], return_tensors="pt", truncation = True, padding='max_length', add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        allEmbedding[i]= last_hidden_states[0][0].detach().numpy()
    del(outputs)
  except:
    pass


with open(FOLDER_NAME+'all_rawBERTembedding.p', 'wb') as fp:
    pickle.dump(allEmbedding,fp)

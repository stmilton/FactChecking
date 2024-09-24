import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from SentenceBert.model_cls import Model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.models.bert import BertTokenizer
from transformers import RobertaTokenizer, RobertaModel

from SentenceBert.data_helper import load_data, SentDataSet, collate_func, convert_token_id
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def get_similar_sentence(claim, evidences, bert_pretrain_path, checkpoint_path):

    tokenizer = BertTokenizer.from_pretrained(bert_pretrain_path)
    model = Model(bert_pretrain_path)
    loss_fct = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_fct.cuda()
    model.load_state_dict(torch.load(checkpoint_path))

    model.eval()
    sent2sim = {}
    for ev_sent in evidences:
        sent2sim[ev_sent] = cosSimilarity(claim, ev_sent, model, tokenizer)
    sent2sim = list(sent2sim.items())
    sent2sim.sort(key=lambda s: s[1], reverse=True)
    print("sent2sim",sent2sim)
    # ev_sent = [s[0] for s in sent2sim[:5] if s[1] > 0.8]
    ev_sent_th08 = [s[0] for s in sent2sim if s[1] > 0.7]
    # print(ev_sent_th08)
    # ev_sent_out5 = [s[0] for s in sent2sim[:5]]
    return ev_sent_th08



def cosSimilarity(sent1, sent2, model, tokenizer):
    model.eval()
    s1_input_ids, s1_mask, s1_segment_id = convert_token_id(sent1, tokenizer)
    s2_input_ids, s2_mask, s2_segment_id = convert_token_id(sent2, tokenizer)

    if torch.cuda.is_available():
        s1_input_ids, s2_input_ids = s1_input_ids.cuda(), s2_input_ids.cuda()

    with torch.no_grad():
        s1_embeddings = model.encode(s1_input_ids, encoder_type='last-avg')
        s2_embeddings = model.encode(s2_input_ids, encoder_type='last-avg')

    cos_sim = F.cosine_similarity(s1_embeddings, s2_embeddings)
    return cos_sim.item()


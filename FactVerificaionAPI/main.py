from fastapi import FastAPI
import torch
import modelUse
from QueryGeneration.QueryGenerator import QueryGenerator
import re
from SentenceBert.SentenceBert import get_similar_sentence
from Prompt.P_tuning_v1 import P_tuning_v1
from crawlerAPI import url_to_text
from googletranslateapi import english_to_chinese, chinese_to_english, get_lang
from SentenceBert.model_cls import Model
from torch import nn
from transformers.models.bert import BertTokenizer

# uvicorn main:app --host 0.0.0.0 --port 80
app = FastAPI()


@app.get("/equery/")
async def api_001(claim: str):
    try:
        isChinese = False
        if get_lang(claim) == 'zh-TW':
            isChinese = True
            claim = chinese_to_english(claim)
        queryGenerator = QueryGenerator("t5", "t5-base")
        queryGenerator.training_args.batch_size = 8
        queryGenerator.training_args.max_seq_length = 64
        queryGenerator.training_args.decoder_max_length = 20
        queryGenerator.model.load_state_dict(torch.load('./checkpoint/queryGenerator.pth'))
        query = queryGenerator.predict([claim])
        query[0] = query[0].replace("-LRB-","(")
        query[0] = query[0].replace("-RRB-",")")
        if isChinese:
            query[0] = query[0].replace("_"," ")
            query[0] = english_to_chinese(query[0])
            
            return {"equery":query[0], "state":"1"}
        return {"equery":query[0], "state":"1"}

    except Exception as e:
        return {"equery":None, "state":e}

@app.get("/everify/")
async def api_002(claim: str, url:str):
    try:
        text = url_to_text([url])[0]
        evidences = re.split(r'(?<=[.?!;])', text)
        evidences = list(evidence for evidence in evidences if len(evidence) > 50)
        
        gold_evidences = get_similar_sentence(claim, evidences, 'bert-base-uncased', "./checkpoint/sentenceBert_bert_checkpoint.ckpt")
        if len(gold_evidences) == 0:
            return {"label":2, "evidence":gold_evidences, "state":"1"}
        verifi_model = P_tuning_v1("bert","bert-base-uncased")
        verifi_model.model.load_state_dict(torch.load("./checkpoint/p_tuning_SentenceBert_out5_bert.pth"))

        most_label, most_evidences = verifi_model.predict(claim, gold_evidences)
        return {"label":most_label, "evidence":most_evidences, "state":"1"}

    except Exception as e:
       return {"label":None, "evidence":None, "state":e}

@app.get("/cverify/")
async def api_003(claim: str, url:str):
    try:            
        text = url_to_text([url])[0]

        isChinese = False
            
        if get_lang(claim) == 'zh-TW':
            
            isChinese = True
            claim = chinese_to_english(claim)
        
        if get_lang(text) == 'zh-TW':
            # 過濾非中英文
            chinese_english_pattern = re.compile('[a-zA-Z\u4e00-\u9fa5]+')
            text = ''.join(chinese_english_pattern.findall(text))

            zh_evidences = re.split(r'[？：。！（）.“”…\t\n]', text)
            zh_evidences = [evidence for evidence in zh_evidences if evidence != '' and len(evidence) > 5]
            
            evidences=[]
            for zh_evidence in zh_evidences:
                evidences.append(chinese_to_english(zh_evidence))
        else:
            evidences = re.split(r'(?<=[.?!;])', text)
            evidences = list(evidence for evidence in evidences if len(evidence) > 50)

        gold_evidences = get_similar_sentence(claim, evidences, 'bert-base-uncased', "./checkpoint/sentenceBert_bert_checkpoint.ckpt")
        
        
        if len(gold_evidences) == 0:
            return {"label":2, "evidence":gold_evidences, "state":"1"}
        verifi_model = P_tuning_v1("bert","bert-base-uncased")
        verifi_model.model.load_state_dict(torch.load("./checkpoint/p_tuning_SentenceBert_out5_bert.pth"))
        
        most_label, most_evidences = verifi_model.predict(claim, gold_evidences)
        if isChinese:
            chinese_most_evidences=[]
            for most_evidence in most_evidences:
                most_evidence = most_evidence.replace("_"," ")
                chinese_most_evidences.append(english_to_chinese(most_evidence))
            most_evidences =  chinese_most_evidences
        return {"label":most_label, "evidence":most_evidences, "state":"1"}

    except Exception as e:
        return {"label":None, "evidence":None, "state":e}
    


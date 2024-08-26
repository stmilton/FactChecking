from fastapi import FastAPI
import torch
import modelUse
from QueryGeneration.QueryGenerator import QueryGenerator
import re
from SentenceBert.SentenceBert import get_similar_sentence
from Prompt.P_tuning_v1 import P_tuning_v1
from crawlerAPI import url_to_text
from googletranslateapi import english_to_chinese, chinese_to_english, get_lang

# uvicorn main:app --host 0.0.0.0 --port 80

# everify

# claim = "Jackie (2016 film) was directed by Peter Jackson."
# url = ["https://en.wikipedia.org/wiki/Jackie_(2016_film)"]
# text = url_to_text(url)[0]
# print(1, text)
# evidences = re.split(r'(?<=[.?!;])\s+', text)
# evidences = re.split(r'(?<=[.?!;])', text)
# evidences = list(evidence for evidence in evidences if len(evidence) > 50)
# print(2, evidences)
# gold_evidences = get_similar_sentence(claim, evidences, 'bert-base-uncased', "FactVerificaionAPI\checkpoint\sentenceBert_bert_checkpoint.ckpt")
# print(3, gold_evidences)
# verifi_model = P_tuning_v1("bert","bert-base-uncased")
# verifi_model.model.load_state_dict(torch.load("FactVerificaionAPI\checkpoint\p_tuning_SentenceBert_out5_bert.pth"))
# most_label, most_evidences = verifi_model.predict(claim, gold_evidences)
# print(4, most_label)
# print(5, most_evidences)


# equery

# claim = "The World According to Paris starred Hilton's then-girlfriend."
# queryGenerator = QueryGenerator("t5", "t5-base")
# queryGenerator.training_args.batch_size = 8
# queryGenerator.training_args.max_seq_length = 64
# queryGenerator.training_args.decoder_max_length = 20
# queryGenerator.model.load_state_dict(torch.load('./FactVerificaionAPI/checkpoint/queryGenerator.pth'))
# query = queryGenerator.predict([claim])
# print("------------")
# print(query[0])
# print("------------")



# cverify

# claim = "新疆棉花生产已实现高度机械化，不需要强迫劳动。"
# url = "https://www.piyao.org.cn/2021-03/25/c_1211081883.htm"

# claim = "截至2020年3月底，猪肉价格持续小幅下跌。"
# url = "https://www.piyao.org.cn/2020-04/20/c_1210578194.htm"

# text = url_to_text([url])[0]
# # print(1, text)
# zh_evidences = re.split(r'[？：。！（）.“”…\t\n]', text)
# zh_evidences = [evidence for evidence in zh_evidences if evidence != '' and len(evidence) > 5]
# # print(2, zh_evidences)
# gold_evidences = get_similar_sentence(claim, zh_evidences, './FactVerificaionAPI/checkpoint/roberta_pretrain', "FactVerificaionAPI\checkpoint\sentenceBert_bert_chinese.ckpt")
# print(3, gold_evidences)


# claim = "新疆棉花生产已实现高度机械化，不需要强迫劳动。"
# evidence = ["截至2020年3月底，猪肉价格持续大幅下跌。"]
# gold_evidences = get_similar_sentence(claim, evidence, 'bert-base-chinese', "FactVerificaionAPI\checkpoint\sentenceBert_bert_chinese.ckpt")
url = "https://zh.wikipedia.org/zh-tw/%E4%B9%A0%E8%BF%91%E5%B9%B3"
text = url_to_text([url])[0]
print(text)
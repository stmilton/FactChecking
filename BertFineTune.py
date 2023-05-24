from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, EarlyStoppingCallback
import numpy as np
import sklearn.metrics as sm
import torch

class BertFineTune():
    def __init__(self,checkpoint, inference = False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

        if not inference:
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            self.training_args = TrainingArguments( output_dir = "./temp", 
                                                    evaluation_strategy= "steps", 
                                                    eval_steps = 50, 
                                                    save_total_limit = 5,
                                                    report_to = "none", 
                                                    load_best_model_at_end = True,
                                                    metric_for_best_model = 'precision',
                                                    )

            self.trainer = Trainer(
                            self.model,
                            self.training_args,
                            # train_dataset=self.datasets["train"],
                            # eval_dataset=self.datasets["validation"],
                            data_collator=self.data_collator,
                            tokenizer=self.tokenizer,
                            compute_metrics=self.compute_metrics,
                            callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
                            )
    def datasets_process(self,datasets):
        self.datasets = datasets

        # 串聯文字
        self.datasets = self.datasets.map(self.concatenated_text)

        # 截斷過長文字
        self.datasets = self.datasets.map(self.truncation_function, batched=True)

        if hasattr(self,"trainer"):
            self.trainer.train_dataset = self.datasets["train"]
            self.trainer.eval_dataset = self.datasets["validation"]

    # 串聯title、message、anchor文字
    def concatenated_text(self, x):
        if x["title"] is not None:
            x["message"] = x["title"] + "[SEP]" + x["message"]
        if x["anchor"] is not None:
            x["message"] = x["message"] + "[SEP]" + x["anchor"]
        return x

    # 截斷過長的部分
    def truncation_function(self, example):
        return self.tokenizer(example["message"], truncation=True)

    # 計算評估數值
    def compute_metrics(self,eval_preds):
        pred, labels = eval_preds
        pred = np.argmax(pred, axis=-1)
        precision, recall, macro_f1, _ = sm.precision_recall_fscore_support(y_true=labels, y_pred=pred, average='macro')
        micro_f1 = sm.f1_score(y_true=labels, y_pred=pred, average='micro')
        return {"precision": precision, "recall": recall, "macro_f1": macro_f1, "micro_f1": micro_f1}
    
    def set_device(self):
        # 使用GPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
    
    def train(self):
        self.trainer.train()

    def label_predict(self,data_part='test'):
        predictions = self.trainer.predict(self.datasets[data_part])
        return predictions.label_ids, [np.argmax(x) for x in predictions.predictions]
    
    def predictions(self):
        # 使用GPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        predictions = []
        scores=[]
        for i in range(len(self.datasets)):
            input_ids = self.datasets[i]['input_ids']
            attention_mask = self.datasets[i]['attention_mask']
            with torch.no_grad():
                output = self.model(input_ids=torch.tensor(input_ids, device=device).unsqueeze(0), attention_mask=torch.tensor(attention_mask, device=device).unsqueeze(0))
                logits = output.logits
                # print(torch.softmax(logits, dim=-1))
                scores.append(torch.softmax(logits, dim=-1).tolist()[0])
                predictions.append(logits.argmax().item())
        return predictions, scores
        

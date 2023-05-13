from typing import Any
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from openprompt.plms import load_plm
import torch
from torch.optim import AdamW
from tool.EarlyStopping import EarlyStopping
from tool.Evaluator import Evaluator

class Prompt():
    def __init__(self, model_name ,checkpoint, inference = False):
        # There are 2 classes in Sentiment Analysis
        self.classes = ["yes", "no", "maybe"]
        self.plm, self.tokenizer, self.model_config, self.WrapperClass = load_plm(model_name, checkpoint)
        self.model = None
        self.template = None
        self.verbalizer = None
        self.training_args = Arguments()
        self.val_evaluator = Evaluator()

    def set_datasets(self,datasets):
        label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        temp_input_examples = []
        for data in datasets:
            temp_evidence = []
            evidence_str = ""
            for evidence in data['evidence']:
                if len(evidence) == 1:
                    temp_evidence.append(evidence[0])
                else:
                    temp_evidence.append(evidence[2])
                evidence_str = ",".join(temp_evidence)
            input_example = InputExample(text_a = evidence_str, 
                                         text_b = data['claim'], 
                                         label=label_mapping[data['label']]) 
            temp_input_examples.append(input_example)
        return temp_input_examples
    
    def set_data_loader(self,input_examples):
        data_loader = PromptDataLoader(dataset=input_examples,
            tokenizer=self.tokenizer,
            template=self.template,
            tokenizer_wrapper_class=self.WrapperClass,
            batch_size=self.training_args.batch_size,
            max_seq_length=self.training_args.max_seq_length, decoder_max_length=3
        )
        return data_loader

    def datasets_process(self,datasets):
        # 定義資料集
        train_input_examples = self.set_datasets(datasets["train"])
        val_input_examples = self.set_datasets(datasets["validation"])
        test_input_examples = self.set_datasets(datasets["test"])
        
        # 定義DataLoader
        train_data_loader = self.set_data_loader(train_input_examples)
        val_data_loader = self.set_data_loader(val_input_examples)
        test_data_loader = self.set_data_loader(test_input_examples)

        self.data_loader = {"train":train_data_loader,"validation":val_data_loader, "test":test_data_loader}
    
    def zero_shot(self):
        self.model.eval()
        print("zero-shot測試")
        with torch.no_grad():
            allpreds = []
            alllabels = []
            for i, batch in enumerate(self.data_loader["test"]):
                if self.training_args.use_cuda:
                    batch = batch.cuda()
                logits = self.model(batch)
                labels = batch['label']
                alllabels.extend(labels.cpu().tolist())
                allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            return alllabels, allpreds

    def train(self):
        if self.training_args.use_cuda:
            self.model = self.model.cuda()
        # 對偏差和 LayerNorm 參數設定不衰减是一個好經驗
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learn_rate, weight_decay=self.training_args.weight_decay)
        loss_func = torch.nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(self.training_args.early_stopping_patience)

        for epoch in range(self.training_args.epoch):
            tot_loss = 0
            self.model.train()
            for step, batch in enumerate(self.data_loader["train"]):
                if self.training_args.use_cuda:
                    batch = batch.cuda()
                logits = self.model(batch)
                labels = batch['label']
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step %300 ==0 and step!=0:
                    print("Epoch {}, step {}, average loss: {}".format(epoch+1, step, tot_loss/step), flush=True)

            # 驗證
            self.model.eval()
            val_preds = []
            val_labels = []
            for step, batch in enumerate(self.data_loader["validation"]):
                if self.training_args.use_cuda:
                    batch = batch.cuda()
                valid_logits = self.model(batch)
                labels = batch['label']
                val_labels.extend(labels.cpu().tolist())
                val_preds.extend(torch.argmax(valid_logits, dim=-1).cpu().tolist())
            
            print("--------validation---------")
            self.val_evaluator.labels = val_labels
            self.val_evaluator.preds = val_preds
            self.val_evaluator.matrixes()
            self.val_evaluator.accuracy()
            precision, recall, macro_f1, micro_f1 = self.val_evaluator.precision_recall_fscore()

            early_stopping(micro_f1*-1, self.model, self.training_args.staging_point)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        self.model.load_state_dict(torch.load(self.training_args.staging_point))
    
    def label_predict(self,data_part='test'):
        # 測試
        allpreds = []
        alllabels = []
        for step, batch in enumerate(self.data_loader[data_part]):
            self.model.eval()

            if self.training_args.use_cuda:
                batch = batch.cuda()
            logits = self.model(batch)
            labels = batch['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return alllabels, allpreds

# 參數預設值
class Arguments():
    def __init__(self):
        self.use_cuda = False
        self.learn_rate = 1e-4
        self.batch_size = 1
        self.epoch = 3
        self.weight_decay = 0
        self.early_stopping_patience = 2
        self.max_seq_length = 512
        self.staging_point = 'checkpoint.pth'
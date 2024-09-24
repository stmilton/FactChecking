from tqdm import tqdm
from Prompt.Prompt import Prompt
from openprompt.prompts import ManualTemplate, PtuningTemplate
from openprompt import PromptForGeneration, PromptDataLoader
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from torch.optim import AdamW
from Prompt.EarlyStopping import EarlyStopping
import torch
import json
from datasets import DatasetDict, load_dataset,Dataset
from openprompt.utils.metrics import generation_metric


class QueryGenerator(Prompt):
    def __init__(self, model_name, checkpoint, inference=False):
        super().__init__(model_name, checkpoint, inference)
        # self.template = ManualTemplate(
        #                     tokenizer=self.tokenizer,
        #                     text="If you want to search for {'placeholder':'text_a', 'shortenable':'True'} on Wikipedia, you should search Wikipedia's page on {'special': '<eos>'} {'mask'}",
        #                 )
        self.template = PtuningTemplate(
                            model=self.plm,
                            tokenizer=self.tokenizer,
                            text='{"soft":"If you want to search for"} {"placeholder":"text_a", "shortenable":"True"} {"soft":"on Wikipedia, you should search Wikipedia\'s page on"} {"special": "<eos>"} {"mask"}',
                        )
        self.model = PromptForGeneration(
                            template=self.template,
                            plm=self.plm,
                            tokenizer= self.tokenizer,
                            freeze_plm = False,
                            plm_eval_mode=False
                        )
        
    def datasets_process(self,datasets):
        train_input_examples = self.data_helper(datasets["train"])
        val_input_examples = self.data_helper(datasets["validation"])

        train_data_loader = self.set_data_loader(train_input_examples)
        val_data_loader = self.set_data_loader(val_input_examples)
        self.data_loader = {"train":train_data_loader,"validation":val_data_loader}

    def data_helper(self, datas):
        temp_input_examples = []
        for data in datas:
            if  data['verifiable'] == "NOT VERIFIABLE":
                continue
            temp_label = set()
            for evidences in data['evidence']:
                for evidence in evidences:
                    # if evidence[2] is not None:
                        temp_label.add(evidence[2])
            # print(",".join(temp_label))
            input_example = InputExample(text_a = data['claim'], 
                                tgt_text=" [SEP] ".join(temp_label))
            temp_input_examples.append(input_example)
        return temp_input_examples
    
    def set_data_loader(self,input_examples):
        data_loader = PromptDataLoader(dataset=input_examples,
            tokenizer=self.tokenizer,
            template=self.template,
            tokenizer_wrapper_class=self.WrapperClass,
            batch_size=self.training_args.batch_size,
            max_seq_length=self.training_args.max_seq_length,
            decoder_max_length=self.training_args.decoder_max_length,
            shuffle=True, 
            teacher_forcing=True, 
            predict_eos_token=True,
            truncate_method="head"
        )
        return data_loader
    def train(self):
        # 對偏差和 LayerNorm 參數設定不衰减是一個好經驗
        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.training_args.learn_rate, weight_decay=self.training_args.weight_decay)
        early_stopping = EarlyStopping(self.training_args.early_stopping_patience)
        self.set_device()

        for epoch in range(self.training_args.epoch):
            tot_loss = 0
            self.model.train()
            for step, batch in enumerate(self.data_loader["train"]):
                if self.training_args.use_cuda:
                    batch = batch.cuda()
                loss = self.model(batch)
                # print(loss)
                loss.backward()

                tot_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                if step %1000 ==0 and step!=0:
                    # 驗證
                    self.model.eval()
                    val_tot_loss = 0
                    
                    for val_step, batch in enumerate(self.data_loader["validation"]):
                        if self.training_args.use_cuda:
                            batch = batch.cuda()
                        valid_loss = self.model(batch)
                        val_tot_loss += valid_loss.item()
                    print("Epoch {}, step {}, average loss: {}, validation loss: {}".format(epoch+1, step, tot_loss/step, val_tot_loss/val_step), flush=True)

                    early_stopping(val_tot_loss, self.model, self.training_args.staging_point)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            if early_stopping.early_stop:
                break
        self.model.load_state_dict(torch.load(self.training_args.staging_point))

    def set_test(self,claims):
        temp_input_examples = []
        for data in claims:
            input_example = InputExample(text_a = data,
                                         tgt_text=""
                                         )
            temp_input_examples.append(input_example)

        test_data_loader = PromptDataLoader(dataset=temp_input_examples,
            tokenizer=self.tokenizer,
            template=self.template,
            tokenizer_wrapper_class=self.WrapperClass,
            batch_size=self.training_args.batch_size,
            max_seq_length=self.training_args.max_seq_length,
            decoder_max_length=self.training_args.decoder_max_length,
            shuffle=False, 
            teacher_forcing=False, 
            predict_eos_token=True,
            truncate_method="head"
        )
        return test_data_loader
    
    def predict(self, claims):
        test_data_loader = self.set_test(claims)
        # print(test_data_loader.wrapped_dataset)
        # self.model.load_state_dict(torch.load(self.training_args.staging_point))

        generated_query = []
        # groundtruth_claim = []
        generation_arguments = {
            "max_length": self.training_args.decoder_max_length,
            "num_return_sequences": 1,
        }
        self.set_device()

        self.model.eval()
        for batch in tqdm(test_data_loader, desc="generating"):
            if self.training_args.use_cuda:
                batch = batch.cuda()
            # print(batch)
            _, output_sentence = self.model.generate(batch, **generation_arguments)
            generated_query.extend(output_sentence)
            # groundtruth_claim.extend(batch['tgt_text'])
        # score = generation_metric(generated_sentence, groundtruth_claim, "sentence_bleu")
        # print("test_score", score, flush=True)
        return generated_query
    

import os

import torch
from BertFineTune import BertFineTune
from datasets import load_dataset
from Prompt.HardPrompt import HardPrompt
from Hyperparameters import Hyperparameters
from tool.Evaluator import Evaluator
from tool.LeaningCurveDrawer import LeaningCurveDrawer
from peft import get_peft_model, LoraConfig, TaskType

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
hyperparameters = Hyperparameters()

# Load Data
def load_data(train_size = None):
    raw_dataset = {}
    if train_size is not None:
        raw_dataset['train'] = load_dataset('json',data_files="Data/train.jsonl",split={'train': train_size})
    else:
        raw_dataset['train'] = load_dataset('json',data_files="Data/train.jsonl",split="train")
    raw_dataset['validation'] = load_dataset('json',data_files="Data/valid.jsonl",split="train")
    raw_dataset['test'] = load_dataset('json',data_files="Data/test.jsonl",split="train")
    hyperparameters.train_size = len(raw_dataset['train'])
    hyperparameters.val_size = len(raw_dataset['validation'])
    hyperparameters.test_size = len(raw_dataset['test'])
    return raw_dataset

"""
prompt_tune模型
"""
def prompt():
    # Load Data
    datasets = load_data()

    # prompt_tune
    hard_prompt = HardPrompt("bert", hyperparameters.checkpoint)
    hard_prompt.datasets_process(datasets)

    # # peft
    # peft_config = LoraConfig(
    # task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    # )
    # hard_prompt.plm = get_peft_model(hard_prompt.plm, peft_config)

    

    # 設定參數
    hard_prompt.training_args.use_cuda = torch.cuda.is_available()
    hard_prompt.training_args.learn_rate = hyperparameters.learn_rate
    hard_prompt.training_args.epoch = hyperparameters.epoch
    hard_prompt.training_args.weight_decay = hyperparameters.weight_decay
    hard_prompt.training_args.batch_size = hyperparameters.batch_size
    hard_prompt.training_args.max_seq_length = hyperparameters.max_seq_length
    hard_prompt.training_args.staging_point = './temp/hard_prompt_checkpoint.pth'
    # 印出參數
    hyperparameters.print("hard prompt")

    # 是否載入之前模型
    # hard_prompt.model.load_state_dict(torch.load(hard_prompt.training_args.staging_point))
    
    # 訓練
    hard_prompt.train()

    # 預測
    print("--------test---------")
    labels,predic = hard_prompt.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()
    evaluator.save("hard_prompt_final")

"""
fine_tune模型
"""
def fine_tune():
    # Load Data
    datasets = load_data()

    # BertFineTune
    bertFineTune = BertFineTune(hyperparameters.checkpoint)
    bertFineTune.datasets_process(datasets)
    # 設定參數
    bertFineTune.training_args.learning_rate = hyperparameters.learn_rate
    bertFineTune.training_args.num_train_epochs = hyperparameters.epoch
    bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
    bertFineTune.training_args.per_device_train_batch_size = hyperparameters.batch_size
    bertFineTune.training_args.per_device_eval_batch_size = hyperparameters.batch_size
    
    # 印出參數
    hyperparameters.print()

    # 訓練
    bertFineTune.train()
    bertFineTune.trainer.save_model("final_checkpoint.ckpt")

    # 預測
    labels,predic = bertFineTune.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()

"""
畫learn_curve
"""
def learn_curve():

    # 設定training_set
    training_set = [1,50,100,150,200,250,300,350,400]

    evaluator = Evaluator()

    for size in training_set:
        train_size = size

        # Load Data
        datasets = load_data(train_size)

        # BertFineTune
        bertFineTune = BertFineTune(hyperparameters.checkpoint)

        bertFineTune.datasets_process(datasets)
        # 設定參數
        bertFineTune.training_args.learning_rate = hyperparameters.learn_rate
        bertFineTune.training_args.num_train_epochs = hyperparameters.epoch
        bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
        bertFineTune.training_args.per_device_train_batch_size = hyperparameters.batch_size
        bertFineTune.training_args.per_device_eval_batch_size = hyperparameters.batch_size

        # 訓練
        bertFineTune.train()

        # 預測
        labels,predic = bertFineTune.label_predict()
        evaluator.labels = labels
        evaluator.preds = predic
        evaluator.matrixes()
        evaluator.accuracy()
        evaluator.precision_recall_fscore()
    
    # 畫出learn_curve
    leaning_curve_drawer = LeaningCurveDrawer(training_set,[(evaluator,"fine_tune")])
    leaning_curve_drawer.accuracy_curve()
    leaning_curve_drawer.recall_curve()
    leaning_curve_drawer.precision_curve()
    leaning_curve_drawer.macro_f1_curve()

if __name__ == '__main__':
    # fine_tune()
    prompt()
    # learn_curve()
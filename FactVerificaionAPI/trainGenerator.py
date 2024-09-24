import json
from QueryGeneration.QueryGenerator import QueryGenerator


if __name__ == '__main__':
    queryGenerator = QueryGenerator("t5", "t5-base")

    datasets = {}
    with open('Data/ori_fever/train.jsonl', 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        datasets['train'] = data

    with open('Data/ori_fever/shared_task_dev.jsonl', 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
        datasets['validation'] = data

    queryGenerator.training_args.epoch = 20
    queryGenerator.training_args.batch_size = 8
    queryGenerator.training_args.max_seq_length = 64
    queryGenerator.training_args.staging_point = f'./temp/queryGenerator.pth'
    queryGenerator.training_args.decoder_max_length = 20
    queryGenerator.training_args.early_stopping_patience = 50
    queryGenerator.datasets_process(datasets)
    # queryGenerator.set_device()
    queryGenerator.train()
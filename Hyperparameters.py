class Hyperparameters():
    def __init__(self):
        self.checkpoint = "bert-base-uncased"
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.epoch = 30
        self.batch_size = 8
        self.learn_rate = 1e-6
        self.weight_decay = 0
        self.max_seq_length = 256

    def print(self, title=""):
        print(title)
        print("-------------------------")
        print("TRAIN_SIZE :",self.train_size)
        print("VALIDATION_SIZE :",self.val_size)
        print("TEST_SIZE :",self.test_size)    
        print("EPOCH :",self.epoch)
        print("BATCH_SIZE :",self.batch_size)
        print("LEARN_RATE :",self.learn_rate)
        print("WEIGHT_DECAY :",self.weight_decay)
        print("-------------------------")

    
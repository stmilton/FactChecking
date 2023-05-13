import sklearn.metrics as sm
from datetime import timezone,timedelta
import datetime
import json

class Evaluator:
    def __init__(self,labels=None,preds=None) -> None:
        self.labels = labels
        self.preds = preds
        self.matrixes_history = []
        self.accuracy_history = []
        self.precision_history = []
        self.recall_history = []
        self.macro_f1_history = []
        self.micro_f1_history = []

    def matrixes(self):
        matrixes = sm.confusion_matrix(self.labels,self.preds)
        self.matrixes_history.append(matrixes)
        print("confusion matrix:")
        print(matrixes)
        return matrixes
    
    def accuracy(self):
        correct_count = sum([int(i == j) for i, j in zip(self.preds, self.labels)])
        total_data = len(self.preds)
        acc = sum([int(i == j) for i, j in zip(self.preds, self.labels)]) / len(self.preds)
        self.accuracy_history.append(acc)
        print("correct count :", correct_count, "total data :", total_data, "accuracy :", acc)
        return acc
    
    def precision_recall_fscore(self):
        precision, recall, macro_f1, _ = sm.precision_recall_fscore_support(y_true=self.labels, y_pred=self.preds, average='macro')
        micro_f1 = sm.f1_score(y_true=self.labels, y_pred=self.preds, average='micro')
        self.precision_history.append(precision)
        print("precision :", precision)
        self.recall_history.append(recall)
        print("recall :", recall) 
        self.macro_f1_history.append(macro_f1)
        print("macro-f1 :", macro_f1)
        self.micro_f1_history.append(micro_f1) 
        print("micro-f1 :", micro_f1)
        return precision, recall, macro_f1, micro_f1
    
    def save(self, file_name = "final_save_"):
        file_name += datetime.datetime.now().astimezone(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M")
        data = {
                "matrixes":self.matrixes_history[-1].tolist(),
                "accuracy":self.accuracy_history[-1],
                "precision":self.precision_history[-1],
                "recall":self.recall_history[-1],
                "macro-f1":self.macro_f1_history[-1],
                "micro-f1":self.micro_f1_history[-1]
                }
        with open(file_name+'.json', 'w') as f:
            json.dump(data, f)
            print("Save Complete")
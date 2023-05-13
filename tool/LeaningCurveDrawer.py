from typing import List,Tuple

from matplotlib import pyplot as plt
from tool.Evaluator import Evaluator


class LeaningCurveDrawer:

    def __init__(self,training_set, evaluator_list: List[Tuple[Evaluator,str]]):
        self.training_set = training_set
        self.evaluator_list = evaluator_list

    def accuracy_curve(self):
        for evaluator,label_name in self.evaluator_list:
            plt.plot(self.training_set, evaluator.accuracy_history, linewidth=2, label=label_name)

            # 寫出數值
            for a,b in zip(self.training_set, evaluator.accuracy_history):
                plt.text(a, b+0.0005, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
        # x軸取log
        plt.xscale("log")
        plt.legend(loc="lower right", fontsize=14)
        plt.xlabel("Training set size take log", fontsize=14)
        plt.ylabel("Test Accuracy", fontsize=14)
        plt.savefig("./output/accuracy_learning_cruve.png") 
        plt.show()
        plt.close()

    def recall_curve(self):
        for evaluator,label_name in self.evaluator_list:
            plt.plot(self.training_set, evaluator.recall_history, linewidth=2, label=label_name)

            # 寫出數值
            for a,b in zip(self.training_set, evaluator.recall_history):
                plt.text(a, b+0.0005, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
        # x軸取log
        plt.xscale("log")
        plt.legend(loc="lower right", fontsize=14)
        plt.xlabel("Training set size take log", fontsize=14)
        plt.ylabel("Test Recall", fontsize=14)
        plt.savefig("./output/recall_learning_cruve.png") 
        plt.show()
        plt.close()

    def precision_curve(self):
        for evaluator,label_name in self.evaluator_list:
            plt.plot(self.training_set, evaluator.precision_history, linewidth=2, label=label_name)

            # 寫出數值
            for a,b in zip(self.training_set, evaluator.precision_history):
                plt.text(a, b+0.0005, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
        # x軸取log
        plt.xscale("log")
        plt.legend(loc="lower right", fontsize=14)
        plt.xlabel("Training set size take log", fontsize=14)
        plt.ylabel("Test Precision", fontsize=14)
        plt.savefig("./output/precision_learning_cruve.png") 
        plt.show()
        plt.close()

        
    def macro_f1_curve(self):
        for evaluator,label_name in self.evaluator_list:
            plt.plot(self.training_set, evaluator.macro_f1_history, linewidth=2, label=label_name)

            # 寫出數值
            for a,b in zip(self.training_set, evaluator.macro_f1_history):
                plt.text(a, b+0.0005, '%.3f' % b, ha='center', va= 'bottom',fontsize=7)
        # x軸取log
        plt.xscale("log")
        plt.legend(loc="lower right", fontsize=14)
        plt.xlabel("Training set size take log", fontsize=14)
        plt.ylabel("Test macro_f1", fontsize=14)
        plt.savefig("./output/macro_f1_learning_cruve.png") 
        plt.show()
        plt.close()
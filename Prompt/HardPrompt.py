from Prompt.Prompt import Prompt
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptForClassification

class HardPrompt(Prompt):
    def __init__(self, model_name, checkpoint, inference=False):
        super().__init__(model_name, checkpoint, inference)
        self.template = ManualTemplate(
                            tokenizer=self.tokenizer,
                            text='{"placeholder":"text_a", "shortenable":True} According to the above content, the following description: {"placeholder":"text_b", "shortenable":True} whether it is correct? please answer, yes or no or maybe. {"mask"}.',
                        )
        
        self.verbalizer = ManualVerbalizer(
                            classes=self.classes,
                            num_classes=3,
                            label_words=[["yes", "right"], ["no"], ["maybe"]],
                            tokenizer=self.tokenizer,
                        )
        self.model = PromptForClassification(
                            template=self.template,
                            plm=self.plm,
                            verbalizer=self.verbalizer,
                            freeze_plm = False
                        )
        

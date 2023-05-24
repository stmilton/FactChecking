from transformers import BartForConditionalGeneration, \
                         T5Config, T5Tokenizer, T5ForConditionalGeneration
from openprompt.plms.seq2seq import T5LMTokenizerWrapper, T5TokenizerWrapper
                    

class InstructDial():
    def __init__(self, model_name ,checkpoint, inference = False):

        if model_name == "t5":
            modelclass = {
                'config': T5Config,
                'tokenizer': T5Tokenizer,
                'model': T5ForConditionalGeneration,
            }
        elif model_name == 'bart':
            modelclass = {
                'config': T5Config,
                'tokenizer': T5Tokenizer,
                'model': BartForConditionalGeneration,
            }
        self.model = 
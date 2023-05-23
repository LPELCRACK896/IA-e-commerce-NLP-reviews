
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.nn as nn
import pandas as pd
import torch
from sklearn.metrics import r2_score, accuracy_score

class BERT():

    def __init__(self, tokenizer, model, data, target, evaluation_target, target_type) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.data: pd.DataFrame = data
        self.target = target
        self.evaluation_target = evaluation_target
        self.target_type = target_type


    def the_r2_score(self, data: pd.DataFrame, y_pred_name: str, y_real_name: str):
        return r2_score(data[y_real_name].tolist(), data[y_pred_name].tolist())
    
    def the_acuraccy_score(self, data: pd.DataFrame, y_pred_name: str, y_real_name: str):
        return  accuracy_score(data[y_real_name].tolist(), data[y_pred_name].tolist())
    
    def get_text_value(self, text)-> int:
        tokens_tensors =  self.tokenize(text)
        results = []
        for tokens in tokens_tensors:
            result = self.model(**tokens)
            results.append(int(torch.argmax(result.logits)))
        return int(sum(results) / len(results)) + 1

    def evaluate_good_bad(self, text)-> int: 
        return 1 if self.get_text_value(text)>=3 else 0 
    
    def tokenize(self, text): # Uses sliding window
        full_tokenized_text = self.tokenizer.tokenize(text)
        
        if len(full_tokenized_text) <= 512:
            return [self.tokenizer.encode_plus(" ".join(full_tokenized_text), padding='max_length', max_length=512, truncation=True, return_tensors='pt')]

        subtexts = []
        start = 0
        end = 512
        overlap = 50
        while start < len(full_tokenized_text):
            subtexts.append(full_tokenized_text[start:end])
            start = end - overlap
            end = start + 512
            
        subtexts_tensors = [self.tokenizer.encode_plus(" ".join(subtext), padding='max_length', max_length=512, truncation=True, return_tensors='pt') for subtext in subtexts]
        return subtexts_tensors

    def simple_prediction(self) -> pd.DataFrame:
        data: pd.DataFrame = self.data
        functions_dict = {
            'sentiment': self.evaluate_good_bad,
            'rate': self.get_text_value
        }
        if self.target_type not in functions_dict:
            print(f"Unable to make simple predictio due evaluation_target type {self.target_type}")
            return
        
        data[self.target_type] = data[self.target].apply(lambda text: functions_dict[self.target_type](text))

        if self.target_type == "rate":
            accuracy = self.the_acuraccy_score(data, self.target_type, self.evaluation_target)
            r2 = self.the_r2_score(data, self.target_type, self.evaluation_target)
            print(f'Accuracy: {accuracy * 100}%')
            print(f'R^2: {r2}')

        elif self.target_type == "sentiment":
            accuracy = self.the_acuraccy_score(data, self.target_type, self.evaluation_target)
            print(f'Accuracy: {accuracy * 100}%')


    
    def train(self, model, dataloader, optimizer, scheduler, device):
        model.train()

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)

            loss = outputs.loss

            loss.backward()

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()

        return model


if __name__ == "__main__": 
    bert = BERT()
    text = 'Some major design flaws. I had such high hopes for this dress and really wanted it to work for me. i initially ordered the petite small (my usual size) but i found this to be outrageously small. so small in fact that i could not zip it up! i reordered it in petite medium, which was just ok. overall, the top half was comfortable and fit nicely, but the bottom half had a very tight under layer and several somewhat cheap (net) over layers. imo, a major design flaw was the net over layer sewn directly into the zipper - it c'
    tkn2 = bert.get_text_value(text)
    print(1)

from utils import clean_cr_dataset_on_Rating, clean_cr_dataset_on_Recommended_IND, pre_proccess_data
from BERT import BERT
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

data = pd.read_csv('./Clothing_review/w_reviews.csv', index_col=0)
data = pre_proccess_data(data)

data_ra = clean_cr_dataset_on_Rating(data)
data_ra = data_ra.sample(n=20)
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
bert_ra = BERT(tokenizer = tokenizer, model = model, data = data_ra, target="Text", evaluation_target="Rating", target_type="rate")
bert_ra.simple_prediction()
print('a')
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
def pre_proccess_data(data):
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

    data.dropna(subset=['Title', 'Review Text'], inplace=True)
    data['Text'] = data['Title'].apply(str) + '. ' + data['Review Text'].apply(str)
    data['Text Length'] = data['Text'].apply(lambda x: len(str(x)))
    data['Text Length Token'] = data['Text'].apply(lambda text: len(tokenizer.tokenize(text)))
    return data

def random_delete(dataset: pd.DataFrame, num_filas: int, column: str, value):

    to_delete = dataset[dataset[column] == value]

    if num_filas > len(to_delete):
        raise Exception('El número de filas para eliminar es mayor que las filas disponibles con ese valor.')

    indices_to_delete = np.random.choice(to_delete.index, num_filas, replace=False)

    dataset = dataset.drop(indices_to_delete)

    return dataset

def clean_cr_dataset_on_Recommended_IND(data):
    recommended_counts = data['Recommended IND'].value_counts()
    positivo = recommended_counts[1]
    negativo = recommended_counts[0]
    data = random_delete(data, positivo-negativo, 'Recommended IND', 1)
    return data



def clean_cr_dataset_on_Rating(data):
    rating_counts = dict(data['Rating'].value_counts())
    min_rate_count = min(rating_counts.values())
    for rate, count in rating_counts.items():
        diff = count - min_rate_count
        if diff>0:
            data = random_delete(data, diff, 'Rating', rate)
    return data



import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')

# Cargar el csv
df = pd.read_csv('Train.csv')

# Mapeo de sentimientos
sentiment_mapping = {
    0: 'Cannot Say',
    1: 'Negative',
    2: 'Positive',
    3: 'No Sentiment'
}

# Reemplazar los números por los nombres en la columna de Sentimientos
df['Sentiment'] = df['Sentiment'].replace(sentiment_mapping)

# Imprimir las primeras filas para revisar los datos
print(df.head())

# Imprimir la información básica del dataframe
print(df.info())

# Descripción estadística del dataframe
print(df.describe())

# Contar cuántos de cada tipo de sentimiento hay
print(df['Sentiment'].value_counts())

# Visualización de la distribución de sentimientos
sns.countplot(x='Sentiment', data=df)
plt.show()

# Análisis de Product_Type
print(df['Product_Type'].value_counts())
sns.countplot(x='Product_Type', data=df)
plt.show()

# Preprocesamiento de texto
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return " ".join(filtered_sentence)

df['Processed_Description'] = df['Product_Description'].apply(preprocess_text)

print(df.head())

# Agrupar por tipo de producto y sentimiento
product_sentiment = df.groupby(['Product_Type', 'Sentiment']).size().unstack()

# Gráfico de barras apiladas
product_sentiment.plot(kind='bar', stacked=True, figsize=(10,7))
plt.title('Distribución de sentimientos por tipo de producto')
plt.ylabel('Cantidad de comentarios')
plt.xlabel('Tipo de producto')
plt.show()

# Crear una lista con todas las palabras de los comentarios positivos
positive_comments = df[df['Sentiment'] == 'Positive']['Processed_Description'].str.split().tolist()
positive_comments = [word for sublist in positive_comments for word in sublist]

# Crear una lista con todas las palabras de los comentarios negativos
negative_comments = df[df['Sentiment'] == 'Negative']['Processed_Description'].str.split().tolist()
negative_comments = [word for sublist in negative_comments for word in sublist]

# Contar las palabras más comunes en los comentarios positivos
positive_counter = Counter(positive_comments)
print("Palabras más comunes en comentarios positivos:")
for word, count in positive_counter.most_common(10):
    print(f"{word}: {count}")

# Contar las palabras más comunes en los comentarios negativos
negative_counter = Counter(negative_comments)
print("Palabras más comunes en comentarios negativos:")
for word, count in negative_counter.most_common(10):
    print(f"{word}: {count}")

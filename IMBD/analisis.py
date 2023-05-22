import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train.csv')

print(df.head())
print(df.info())
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)
plt.figure(figsize=(6, 4))
sns.countplot(x='sentiment', data=df)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Distribution of Sentiments')
plt.show()
df['text_length'] = df['text'].apply(len)
print(df['text_length'].describe())
plt.figure(figsize=(8, 6))
sns.histplot(df['text_length'], bins=20)
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.title('Distribution of Text Lengths')
plt.show()

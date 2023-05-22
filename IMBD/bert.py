import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

# Preparación del conjunto de datos
data = pd.read_csv('train.csv')  # Reemplaza 'ruta_del_archivo.csv' con la ubicación de tu archivo CSV

# Limpieza de texto
def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)  # Eliminar caracteres especiales y números
    text = text.lower()  # Convertir a minúsculas
    text = text.split()  # Tokenización
    text = [word for word in text if word not in set(stopwords.words('english'))]  # Eliminar stopwords
    text = ' '.join(text)
    return text

data['text'] = data['text'].apply(clean_text)

# Codificación de etiquetas de sentimiento
label_mapping = {'neg': 0, 'pos': 1}
data['sentiment'] = data['sentiment'].map(label_mapping)

# División del conjunto de datos
train_data, test_data, train_labels, test_labels = train_test_split(data['text'], data['sentiment'], test_size=0.2, random_state=42)

# Preparación de los datos para BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_data), truncation=True, padding=True)
test_encodings = tokenizer(list(test_data), truncation=True, padding=True)

# Construcción del modelo BERT para clasificación
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Entrenamiento del modelo
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
train_inputs = torch.tensor(train_encodings['input_ids']).to(device)
train_masks = torch.tensor(train_encodings['attention_mask']).to(device)
train_labels = torch.tensor(train_labels.values).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
num_epochs = 5
batch_size = 4  # Tamaño de lote reducido
accumulation_steps = 4  # Número de pasos de acumulación de gradientes
total_loss = 0

model.train()
for epoch in range(num_epochs):
    for i in range(0, len(train_inputs), batch_size):
        optimizer.zero_grad()
        batch_inputs = train_inputs[i:i+batch_size]
        batch_masks = train_masks[i:i+batch_size]
        batch_labels = train_labels[i:i+batch_size]
        outputs = model(input_ids=batch_inputs, attention_mask=batch_masks, labels=batch_labels)
        loss = outputs.loss
        loss = loss / accumulation_steps  # Dividir la pérdida por el número de pasos de acumulación
        loss.backward()
        
        total_loss += loss.item()
        
        if (i+1) % accumulation_steps == 0:  # Realizar una actualización de los pesos después de un cierto número de pasos de acumulación
            optimizer.step()
            optimizer.zero_grad()
            
    # Imprimir la pérdida promedio por época
    average_loss = total_loss / (len(train_inputs) / batch_size)
    print(f"Epoch {epoch + 1}: Average Loss = {average_loss}")
    total_loss = 0

# Evaluación del modelo
model.eval()
test_inputs = torch.tensor(test_encodings['input_ids']).to(device)
test_masks = torch.tensor(test_encodings['attention_mask']).to(device)
test_labels = torch.tensor(test_labels.values).to(device)
with torch.no_grad():
    outputs = model(input_ids=test_inputs, attention_mask=test_masks)
    predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()

accuracy = accuracy_score(test_labels.cpu(), predictions)
precision = precision_score(test_labels.cpu(), predictions)
recall = recall_score(test_labels.cpu(), predictions)
f1 = f1_score(test_labels.cpu(), predictions)

# Análisis de la correlación entre sentimiento y calificación
correlation = data['sentiment'].corr(data['calificación'], method='pearson')

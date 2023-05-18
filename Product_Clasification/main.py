import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords
import re
import seaborn as sns
import matplotlib.pyplot as plt

# Paso 1: Carga y limpieza de datos
df = pd.read_csv('Participants_Data/Sample Submission.csv')

# Eliminar valores nulos
df.dropna(inplace=True)

# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Eliminación de caracteres especiales, números y espacios extra
df['comentario_limpio'] = df['comentario'].apply(lambda x: re.sub(r'[^\w\s]', '', x.lower()))
df['comentario_limpio'] = df['comentario_limpio'].apply(lambda x: re.sub(r'\d+', '', x))
df['comentario_limpio'] = df['comentario_limpio'].apply(lambda x: re.sub(r'\s+', ' ', x))

# Lematización/Stemming y eliminación de palabras vacías - Este paso puede variar dependiendo del idioma y la biblioteca que se utilice

# Paso 2: Análisis Exploratorio de Datos
# Aquí puedes hacer gráficos y otras formas de análisis. Este paso dependerá de tus datos específicos.

# Paso 3: Preprocesamiento para BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)  # O el modelo BERT que elijas
encoded_data = tokenizer.batch_encode_plus(
    df['comentario_limpio'].tolist(),
    add_special_tokens=True,
    return_attention_mask=True,
    pad_to_max_length=True,
    max_length=256,
    return_tensors='tf'
)

# Dividir en conjuntos de entrenamiento y prueba
input_ids = encoded_data['input_ids']
attention_masks = encoded_data['attention_mask']
labels = df['etiqueta'].values  # Asegúrate de que 'etiqueta' es el nombre correcto de tu columna de etiquetas
train_inputs, test_inputs, train_labels, test_labels, train_masks, test_masks = train_test_split(
    input_ids,
    labels,
    attention_masks,
    random_state=42,
    test_size=0.1
)

# Paso 4: Entrenar el modelo
model = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased',  # O el modelo BERT que elijas
    num_labels=2  # Cambia esto si tienes más de dos etiquetas
)
optimizer = Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
history = model.fit(
    [train_inputs, train_masks],
    train_labels,
    batch_size=16,  # Ajusta esto según el tamaño de tu GPU
    epochs=4  # Ajusta esto según tus necesidades
)

# Paso 5: Evaluar el modelo
model_eval = model.evaluate([test_inputs, test_masks], test_labels, batch_size=16)

# Mostrar las métricas de evaluación
print("Loss del modelo: ", model_eval[0])
print("Precisión del modelo: ", model_eval[1])

# También puedes hacer predicciones y ver cómo se ven para algunos ejemplos.
predictions = model.predict([test_inputs, test_masks])

# Paso 6: Interpretación de los resultados
# Convertir las predicciones a etiquetas de clase
predictions_labels = np.argmax(predictions.logits, axis=-1)

# Mostrar las predicciones junto a las etiquetas reales en un DataFrame
results_df = pd.DataFrame({'Predicciones': predictions_labels, 'Etiquetas Reales': test_labels})

# Mostrar las primeras filas del DataFrame
print(results_df.head())

# También puedes usar técnicas como la matriz de confusión para visualizar el rendimiento del modelo
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_labels, predictions_labels)
sns.heatmap(cm, annot=True, fmt='d')

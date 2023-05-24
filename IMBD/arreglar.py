import pandas as pd

# Cargar el dataset original desde un archivo CSV
df = pd.read_csv('train.csv')

# Crear una nueva columna "sentimiento" basada en la columna "sentiment"
df['sentimiento'] = df['sentiment'].apply(lambda x: 1 if x == 'pos' else 0)

# Guardar el dataset modificado en un nuevo archivo CSV
df.to_csv('dataset_modificado.csv', index=False)

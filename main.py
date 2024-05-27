from fastapi import FastAPI, Query
import pandas as pd
import pyarrow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

app.title = "Machine Learning Operations"
app.version = "1.0.0"

df_games = pd.read_parquet('Dataset\df_steam_games.parquet') 
df_items = pd.read_parquet('Dataset\df_user_reviews.parquet')

# Ruta para consultar la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
@app.get("/developer/{desarrollador}")
# Función para analizar los datos
def developer(desarrollador: str):    

    #Cargar los dataframes
    df_games = pd.read_parquet('Dataset/df_steam_games.parquet') 
    df_items = pd.read_parquet('Dataset/df_user_reviews.parquet')
    
    # Filtrar el DataFrame df_games por el desarrollador
    df_filtered = df_games[df_games['developer'] == desarrollador]

    # Crear una columna para indicar si el juego es gratuito (price == 0)
    df_filtered['free_gamers'] = (df_filtered['price'] == 0).astype(int)

    # Agrupar por año y calcular la cantidad de items y el porcentaje de contenido free
    grouped = df_filtered.groupby('year').agg(
        Cantidad_de_Items=('item_id', 'count'),
        Contenido_Free=('free_gamers', lambda x: (x.sum() / len(x)) * 100)
    ).reset_index()

    # Renombrar las columnas para la salida deseada
    grouped.columns = ['Año', 'Cantidad de Items', 'Contenido Free']

    # Convertir el DataFrame agrupado a un diccionario orientado por registros
    result_dict = grouped.to_dict(orient='records')

    return result_dict


# Ruta para aplicar el modelo de recomendacion aplicando la similitud del coseno
@app.get("/recomendacion_juego/{id}")
def recomendacion_juego(id: int):

     #Cargar el dataframe para aplicar el modelo
    data_modelo = pd.read_parquet('Dataset/df_modelo_similitud.parquet') 

    # Primero, vamos a convertir la columna 'item_name' a una representación numérica usando TF-IDF (Frecuencia de Terminos - Freciencia Inversa de Terminos)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data_modelo['item_name'])

    # Luego, vamos a añadir las columnas numéricas a nuestra matriz de características
    features = np.column_stack([tfidf_matrix.toarray(), data_modelo['rating']])#, data_modelo['user_id'],])

    # Verificamos la existencia del Id del juego a establecer simulitud de juegos
    result = data_modelo[data_modelo['item_id'] == id]
    nombre_del_juego=result.iloc[0]['item_name']

    # Reindexamos el DataFrame
    data_aplicativo = data_modelo.reset_index(drop=True)

    # Ahora, calculamos la matriz de similitud de coseno
    similarity_matrix = cosine_similarity(features)

    # Para hacer recomendaciones, puedes buscar los juegos más similares a un juego dado
    juego = data_aplicativo[data_aplicativo['item_id'] == id].index[0]
    score = list(enumerate(similarity_matrix[juego]))
    score= sorted(score, key=lambda x: x[1], reverse=True)
    resultado = score[1:6]
    total = data_modelo['item_name'].iloc[[i[0] for i in resultado]].tolist()
    return {'Juego Recomendado ': total}






 
 



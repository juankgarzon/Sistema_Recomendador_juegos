from fastapi import FastAPI, Query
import pandas as pd
import pyarrow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi.responses import HTMLResponse

app = FastAPI()

app.title = "Machine Learning Operations"
app.version = "1.0.0"

# Ruta para consultar la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
@app.get("/developer/{desarrollador}")
# Función para analizar los datos
def developer(desarrollador: str):    

    #Cargar los dataframes
    df_games = pd.read_parquet('Dataset/df_steam_games.parquet') 
        
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


# Ruta para consultar el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.
@app.get("/UserForGenre/{genero}")
# Función para analizar los datos
def UserForGenre(genero: str):  
     #Cargar los dataframes
    merged_df = pd.read_parquet('Dataset/df_UserForGenre.parquet') 
        
     # Filtrar el DataFrame por el género dado
    genre_df = merged_df[merged_df['genres'] == genero]
    
    # Si el DataFrame filtrado está vacío, retornar el formato esperado con valores nulos
    if genre_df.empty:
        return {"Usuario con más horas jugadas para Género " + genero: None, "Horas jugadas": []}
    
    # Agrupar por usuario y sumar las horas jugadas
    user_hours = genre_df.groupby('id')['early_access'].sum()
    
    # Encontrar el usuario con más horas jugadas
    max_user = user_hours.idxmax()
    
    # Agrupar por año y sumar las horas jugadas
    year_hours = genre_df.groupby('year')['early_access'].sum().reset_index()
    
    # Formatear el resultado en la estructura solicitada
    hours_per_year = [{"Año": row['year'], "Horas": row['early_access']} for _, row in year_hours.iterrows()]
    
    result = {
        "Usuario con más horas jugadas para Género " + genero: max_user,
        "Horas jugadas": hours_per_year
    }
    
    return result

# Ruta para aplicar el modelo de recomendacion aplicando la similitud del coseno
@app.get("/ best_developer_year/{id}")
def  best_developer_year(anio: int):

    # Se carga el dataset para la ejecución del EndPont 
    merged_df = pd.read_parquet('Dataset/df_best_developer_year.parquet') 

    # Filtrar las filas donde 'year' es igual al año dado y 'reviews_recommend' es True
    filtered_df = merged_df[(merged_df['year'] == anio) & (merged_df['reviews_recommend'] == True)]

    # Contar recomendaciones por desarrollador
    developer_counts = filtered_df['developer'].value_counts()

    # Obtener los top 3 desarrolladores
    top_3_developers = developer_counts.head(3)

    # Formatear el resultado como un diccionario
    result = {f"Puesto {i+1}": developer for i, developer in enumerate(top_3_developers.index)}      

    return result


# Ruta para aplicar el modelo de recomendacion aplicando la similitud del coseno
@app.get("/developer_reviews_analysis/{desarrollador}")
def developer_reviews_analysis(desarrollador: str):
    # Se carga el dataset para la ejecución del EndPont 
    merged_df = pd.read_parquet('Dataset/df_reseñas_desarrollador.parquet') 

    # Filtrar por desarrolladora
    filtered_df = merged_df[merged_df['developer'] == desarrollador]

    # Contar las reseñas positivas y negativas
    positive_reviews = filtered_df[filtered_df['reviews_recommend'] == True].shape[0]
    negative_reviews = filtered_df[filtered_df['reviews_recommend'] == False].shape[0]
    # Crear el diccionario de resultados
    return {desarrollador: {'Negative': negative_reviews, 'Positive': positive_reviews}}



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
    
from fastapi import FastAPI, Query
import pandas as pd
import pyarrow

app = FastAPI()

app.title = "Machine Learning Operations"
app.version = "1.0.0"

df_games = pd.read_parquet('Dataset\df_steam_games.parquet') 
df_items = pd.read_parquet('Dataset\df_user_reviews.parquet')

# Ruta para consultar la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora
@app.get("/developer/{desarrollador}")
# Función para analizar los datos
def developer(desarrollador: str):    
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
 
 



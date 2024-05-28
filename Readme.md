# <h1 align=center>**`Sistema de Recomendación de Videojuegos  `**</h1>  
<p align="center">
<img src="ML_image.jpg" height="300">
</p>
Este proyecto simula el rol de un Ingeniero MLOps, combinando las habilidades de un Ingeniero de Datos y un Científico de Datos, enfocado en la plataforma multinacional de videojuegos Steam. Para su desarrollo, se proporcionan datos específicos y se solicita la creación de un Producto Mínimo Viable que incluya una API desplegada en un servicio en la nube y la aplicación de un modelo de Machine Learning para realizar recomendaciones de juegos.

### Contexto
Steam es una plataforma de distribución digital de videojuegos desarrollada por Valve Corporation, lanzada en septiembre de 2003 inicialmente para proveer actualizaciones automáticas a sus juegos. Con el tiempo, se expandió para incluir juegos de terceros. Actualmente, cuenta con más de 325 millones de usuarios y un catálogo de más de 25,000 juegos. Cabe destacar que las cifras publicadas por SteamSpy se actualizan solo hasta 2017, ya que a principios de 2018 Steam limitó el acceso a sus estadísticas, lo que dificulta obtener datos precisos desde entonces.

### Datos Iniciales
Para este proyecto se proporcionaron tres archivos JSON:

user_reviews.json: Este archivo contiene opiniones de los usuarios sobre los juegos que consumen, incluyendo recomendaciones, emoticones de humor y estadísticas sobre la utilidad de los comentarios para otros usuarios. También proporciona el ID del usuario que comenta, su URL de perfil y el ID del juego comentado.
users_items.json: Este archivo detalla los juegos jugados por todos los usuarios, junto con el tiempo total que cada usuario ha dedicado a cada juego específico.
steam_games.json: Este archivo abarca información sobre los juegos en sí, tales como títulos, desarrolladores, precios, especificaciones técnicas, etiquetas y otros detalles.

### ETL (Extracción, Transformación y Carga)
Se llevó a cabo la extracción, transformación y carga de los archivos JSON mencionados anteriormente. Los datos se filtraron y se convirtieron en DataFrames de pandas para realizar las transformaciones pertinentes, como se observa en los archivos con el prefijo "ETL" en este repositorio de GitHub. Inicialmente, se optó por convertir los tres archivos principales a formato CSV para ordenarlos y visualizarlos de manera clara. Posteriormente, según las transformaciones necesarias para los endpoints de la API, se generaron archivos Parquet optimizados para cada endpoint.

### EDA y Modelo de Recomendación
En esta parte del proyecto, se llevó a cabo un Análisis Exploratorio de Datos (EDA) utilizando el conjunto de datos proporcionado. A partir de este análisis, se decidió crear una métrica combinada que integrara las revisiones recomendadas por los usuarios y una nueva columna de análisis de sentimiento. El objetivo fue desarrollar un algoritmo de recomendación basado en la similitud coseno, lo que permitió comparar y evaluar la similitud entre distintos juegos y usuarios basándose en esta métrica combinada.

El algoritmo de similitud coseno es una técnica utilizada para comparar la similitud entre vectores o documentos. En el contexto de este proyecto, se aplicó este algoritmo para calcular la similitud entre los perfiles de los juegos, basándose en la métrica combinada de revisiones recomendadas y análisis de sentimiento. Este algoritmo mide la similitud entre dos entidades comparando la orientación y dirección de los vectores que las representan.

El funcionamiento del algoritmo de similitud coseno se basa en medir el coseno del ángulo entre dos vectores, lo que refleja la similitud entre ellos. En el caso de los juegos, esta similitud ayuda a identificar juegos con perfiles similares según la percepción de los usuarios. Esto permite ofrecer recomendaciones más precisas basadas en las opiniones y sentimientos expresados por los jugadores.

### Desarrollo de la API (FastAPI)
En esta sección, se utilizó FastAPI para desarrollar los endpoints del proyecto, lo que permitió mantener el código bien estructurado y prolijo. Debido a la gran cantidad de datos trabajados, con archivos que en ocasiones llegaban a tener hasta 6 millones de columnas, se decidió crear archivos Parquet para cada endpoint de la API. Esta estrategia optimiza el manejo de datos, permitiendo que la información esté disponible en DataFrames listos para ser filtrados según los datos ingresados en la API, mejorando así la eficiencia en la respuesta.

A continuación, se proporciona información sobre el contenido de cada endpoint:

def developer(desarrollador: str): Cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora.

def UserForGenre(genero: str): Devuelve el usuario que acumula más horas jugadas para el género dado y una lista de la acumulación de horas jugadas por año de lanzamiento.

def best_developer_year(año: int): Devuelve el top 3 de desarrolladores con juegos más recomendados por usuarios para el año dado.

def developer_reviews_analysis(desarrolladora: str): Según el desarrollador, devuelve un diccionario con el nombre del desarrollador como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con un análisis de sentimiento como valor positivo o negativo.
Deployment

Para el despliegue de la API, se optó por utilizar la plataforma Render. Render ofrece un entorno confiable y escalable para alojar aplicaciones, lo que garantiza que nuestra API sea accesible y funcione de manera eficiente.

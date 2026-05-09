from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def vectorization(X_data, y_data):

    """
    Función para conversión de texto a vocabulario TF-IDF.

    **Parámetros**: 
    * X_data: columna/lista de texto
    * y_data: columna/lista de clases

    **Funciones**:
    * División de los datos para train/test
    * Creación de vectorizador y vectorización del texto
    * Print del porcentaje de las clases en cada lista de vectores
    """

    # División de datos
    # stratify=y_data mantiene las proporciones de clases
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data     
    )

    # Creanción de vectorizador
    vectorizer = TfidfVectorizer(min_df=3, max_features=100000, ngram_range=(1,2))

    # Vectorizando los comentarios
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    # Proporciones de clases en datos de texto
    print(f'Total comentarios para entrenamiento: {X_train_v.shape[0]} ({round(X_train_v.shape[0]/len(X_data), 2)*100}% del total)')
    print(f"{round(y_train[y_train == 'positive'].shape[0]/y_train.shape[0], 2)*100}% positivos\n{round(y_train[y_train == 'neutral'].shape[0]/y_train.shape[0], 2)*100}% neutros\n{round(y_train[y_train == 'negative'].shape[0]/y_train.shape[0], 2)*100}% negativos\n")
    print(f'Total comentarios para evaluación: {X_test_v.shape[0]} ({round(X_test_v.shape[0]/len(X_data), 2)*100}% del total)')
    print(f"{round(y_test[y_test == 'positive'].shape[0]/y_test.shape[0], 2)*100}% positivos\n{round(y_test[y_test == 'neutral'].shape[0]/y_test.shape[0], 2)*100}% neutros\n{round(y_test[y_test == 'negative'].shape[0]/y_test.shape[0], 2)*100}% negativos\n")
    
    return X_train_v, X_test_v, y_train, y_test



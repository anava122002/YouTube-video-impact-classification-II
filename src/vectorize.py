from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def vectorization(X_data, y_data):
    # stratify=y es importante: mantiene la proporción de clases en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
    )

    # Creando vectorizador
    # ignora palabras que aparecen < 3 veces
    # límite de vocabulario
    # unigramas y bigramas
    vectorizer = TfidfVectorizer(min_df=3, max_features=100000, ngram_range=(1,2))

    # Vectorizando los comentarios
    # fit_transform para train porque el modelo aprende de estos comentarios: qué palabras existen, qué posición ocupa cada una en el vector, cuál es el IDF de cada una.
    # transform para test porque sólo necesitamos vectorizarlos (en base a lo aprendido para train). Si usamos fit volveríamos a reconstruir el modelo en base a los datos de test
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)

    # usar fit con datos de entrenamiento ensucia el aprendizaje (data leakage)

    print(f'Total comentarios para entrenamiento: {X_train_v.shape[0]} ({round(X_train_v.shape[0]/len(X_data), 2)*100}% del total)')
    print(f"{round(y_train[y_train == 'positive'].shape[0]/y_train.shape[0], 2)*100}% positives\n{round(y_train[y_train == 'neutral'].shape[0]/y_train.shape[0], 2)*100}% neutros\n{round(y_train[y_train == 'negative'].shape[0]/y_train.shape[0], 2)*100}% negativos\n")
    print(f'Total comentarios para evaluación: {X_test_v.shape[0]} ({round(X_test_v.shape[0]/len(X_data), 2)*100}% del total)')
    print(f"{round(y_test[y_test == 'positive'].shape[0]/y_test.shape[0], 2)*100}% positives\n{round(y_test[y_test == 'neutral'].shape[0]/y_test.shape[0], 2)*100}% neutros\n{round(y_test[y_test == 'negative'].shape[0]/y_test.shape[0], 2)*100}% negativos\n")
    
    return X_train_v, X_test_v, y_train, y_test



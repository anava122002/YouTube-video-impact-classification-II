import pandas as pd
import numpy as np
from src.vectorize import vectorization
from src.model import train, evaluation
from sklearn.linear_model import LogisticRegression

def main():

    """
    Función principal.
    * Lee datos desde data/
    * Convierte los datos a vocabulario TF-IDF
    * Entrena y evalúa la Modificación 3 del modelo `LogisticRegression()` (ver `README.md`)
    """

    # Importando datos
    df = pd.read_csv("data/english_comments.csv")

    # Vectorizando datos
    X_train, X_test, y_train, y_test = vectorization(df['Comment'], df['Sentiment'])

    # Model o
    print("----- Modelo con Regresión Logística (Modificación 3) ", "-" * 50, "\n")
    log_clf = LogisticRegression(class_weight={'negative': 2.5, 'neutral': 1.5, 'positive': 1}, C=2.5)
    log_results, log_accuracy = train(log_clf, X_train, X_test, y_train, y_test)
    log_report = evaluation(y_test, log_results)


if __name__ == "__main__":
    main()

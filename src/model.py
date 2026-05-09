import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


def train(model, X_train, X_test, y_train, y_test):
    
    """
    Entrena y calcula la accuracy del modelo dado.

    **Parámetros:**
    * model: modelo a entrenar
    * X_train/X_test: lista de texto vectorizado para entrenar/validar
    * y_train/y_test: lista con clases para entrenar/validar
    """

    # Entrenamiento
    model.fit(X_train, y_train)

    # Validación
    results = model.predict(X_test)

    # Cálculo de accuracy
    accuracy = model.score(X_test, y_test)

    return results, accuracy


def evaluation(y_test, y_pred, target_names = ["Negativo", "Neutro", "Positivo"]):

    """
    Imprime classification_report y matriz de confusión.

    **Parámetros:**
    * y_test: clases reales del texto usado para validación
    * y_pred: resultado de la evaluación
    * target_names: nombre de las clases. Por defecto ["Negativo", "Neutro", "Positivo"]
    """

    # Cálculo de porcentajes de precisión
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    # Cálculo de matriz de confusión
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)

    print(classification_report(y_test, y_pred, target_names=target_names))
    fig, ax = plt.subplots(figsize = (4.5,4.5))
    disp.plot(cmap = "Blues", ax = ax)
    plt.show()

    return report_dict
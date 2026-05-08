import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def train(model, X_train, X_test, y_train, y_test):
    

    model.fit(X_train, y_train)

    results = model.predict(X_test)

    accuracy = model.score(X_test, y_test)

    return results, accuracy

def evaluation(y_test, y_pred, target_names = ["Positivo", "Neutro", "Negativo"]):
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)

    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=target_names)

    print(classification_report(y_test, y_pred, target_names=target_names))
    fig, ax = plt.subplots(figsize = (4.5,4.5))
    disp.plot(cmap = "Blues", ax = ax)
    plt.show()

    return report_dict
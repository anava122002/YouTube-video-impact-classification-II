# Modelos de Clasificación de Lenguaje con Scikit-Learn y Pytorch

Tras entender como funcionan los clasificadores de texto y, concretamente, [aquellos basados en Naive Bayes](https://github.com/anava122002/YouTube-video-impact-classification), en este proyecto pretendo dar un paso más. Para ello:

* Implementaré un modelo análogo al que construí desde cero.

* Evaluaré sus limitaciones y les diferentes propondré soluciones.

* Compararé los modelos Naive Bayes con otros basados en Regresión Logística, Regresión Lineal y RandomForest.

## Estructura del Repositorio

```
YouTube-video-impact-classification-II/
    data/
        english_comments.csv
    src/
        __init__.py
        model.py
        preprocessing.py
        utils.py
        vectorizer.py
    .gitignore
    main.py
    README.md
    requirements.txt
```

## Datos 
Los datos usados para este proyecto se encuentran en el siguiente dataset de Kaggle: [Youtube Comments Dataset](https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset). Del total de comnetarios se han extraído únicamente aquellos en inglés, de modo que tenemos:

* 15567 comentarios.

* 66% positivos, 20% neutros y 14% negativos.

## Objetivos

Este proyecto tiene dos objetivos principales:

* Estudiar las diferentes opciones para construir un modelo de clasificación de sentimiento.

* Implementar diferentes modelos, evaluarlos y compararlos entre ellos.

## Fundamentos Teóricos

A continuación se presentan los aspectos teóricos de cada uno de los modelos usados.

### 1. Modelos basados en Naive Bayes: MultinomialNB vs. ComplementNB

*MultinomialNB* es una implementación de Naive Bayes tal y como se vió en [el anterior repositorio](https://github.com/anava122002/YouTube-video-impact-classification). Volver a usarlo sería un error pues ya se vió que no tiene forma de descompensar el desbalanceo.

*ComplementNB* es una implementación análoga pensada para este tipo de situaciones. En lugar de aprender **P(w | clase)**, aprende **P(w | complemento de la clase)**, es decir, la probabilidad de cada palabra en todos los documentos que no pertenecen a esa clase de modo que, al tener la clase mayoritaria probabilidades mejor estimadas, los elementos de la minoritaria se clasificarán con mayor accuracy.

Un posible problema resultante es justo el contrario al que ya encontramos para *MultinomialNB*: la mayoría de comentarios se clasifican como negativos y casi ninguno como positivo, pero esta situación es más fácil de tratar. Esto se debe a que *MultinomialNB* maximiza la verosimilitud de la clase correcta y *ComplementNB* minimiza la de las clases incorrectas.

### 2. Modelos basados en la Regresión Logística

Supongamos que podemos representar al peso de cada palabra como los coeficientes de una regresión lineal:

**z = w<sub>0</sub> + Σ<sub>i=1</sub>w<sub>i</sub>x<sub>i</sub>**

de modo que podemos aplicar a **z** una transformación sigmoidal obteniendo la función de probabilidad:

**P(y=y<sub>k</sub>|x) = σ(z) = 1 / (1+e<sup>-z</sup>)**

para **y<sub>k</sub> = 0 o 1** y **k = 1, 2, ..., n**, siendo **n** el número de clases posibles. 

Estimamos los pesos mediante máxima verosimilitud, asumiendo que las etiquetas siguen una distribución de Bernoulli. Llamamos función de log-loss a: 

**J(w) = −1/m ​Σ<sub>j=1,...,m</sub> ​ [y<sub>j</sub> ​log(hat(y)<sub>j</sub>​) + (1 − y<sub>j</sub>​) log(1−hat(y)<sub>j</sub>​)]+1/(2C) ​Σ<sub>j=1,...,m</sub> ​w<sup>2</sup><sub>i</sub>​**

siendo **m** el número de ejemplos y **1/(2C) ​Σ<sub>j=1,...,m</sub> ​w<sup>2</sup><sub>i</sub>** una​ penalización a pesos grandes. Minimizando esta función mediante gradiente descendente obtenemos los pesos óptimos.

Puesto que estamos trabajando con tres clases, es neceario recurrir a una alternativa a la función sigmoidal (sino cada clasificador sería independiente), en este caso: 

**P(y=k∣x)​ = e<sup>w<sub>j</sub>x</sup> / ​Σ<sub>i=1,...,K </sub>e<sup>w<sub>k</sub>x</sup>​**

La clase elegida será aquella con la probabilidad más alta: **hat(y) ​= arg<sub>k</sub>max​ w<sub>k</sub>⋅x**.

La función log-loss está diseñada para que penalice mucho las predicciones erróneas. Además, al minimizar el coste total el modelo busca un equilibrio entre ajustarse bien a los datos y mantener los pesos pequeños, de modo que un valor de C pequeño genera una penalización mayor y por tanto fuerza a que los pesos sean más cercanos a 0.

### 3. Modelos basados en la Regresión Lineal

Partimos ahora del hiperplano **wx + b = 0**, donde **w** es el vector normal al hiperplano y **b** el sesgo. La distancia de un punto cualquiera será **|wx<sub>j</sub> + b| / ||w||**.

El problema de clasificación basado en la regresión lineal busca maximizar la distancia entre el hiperplano y los vectores de soporte de cada clase (aquí consideramos sólo dos). 

Para un problema linealmente separable, los puntos estarán correctamente clasificados si su distancia al hiperplano es de al menos 1. En la realidad es necesario introducir una variable de holgura, de modo que el problema de se obtiene el problema de optimización:

**min<sub>w</sub> ||w||<sup>2</sup> / 2 + C ​Σ<sub>j=1,...,m</sub> ξ<sub>j</sub>**

En este caso un C alto penaliza mucho las violaciones, forzando al modelo a clasificar correctamente casi todo a costa de un margen más pequeño.

La formulación anterior es análoga a: 

**J(w) = ||w||<sup>2</sup> / 2 + C ​Σ<sub>j=1,...,m</sub> max(0, 1 - y<sub>j</sub> (wx<sub>j</sub> + b))**

conocida como hinge loss. Cuando el ejemplo está mal clasificado o dentro del margen, el término es positivo y contribuye al coste; en caso contrario **1 - y<sub>j</sub> (wx<sub>j</sub> + b) = 0** y no hay penalización.

### 4. Random Forest

Un árbol de decisión al uso sin restricciones se sobreajusta demasiado, lo que en este caso en un problema. Random Forest, por otra parte, previene el overfitting haciendo uso de:

* **Bagging:** para cada árbol se toma una muestra aleatoria con reemplazamiento de n elementos, siendo n el total de observaciones, de modo que cada árbol "ve" datos diferentes y comete distintos errores. Para predecir, cada árbol vota y gana la clase más votada.

* **Feature sampling:** si hay una feature muy predictiva puede darse el caso en que los errores de cada árbol esten correlacionados. Para evitarlo, el nodo considera solo una muestra aleatoria de √n para hacer el split, haciendo a los árboles más distintos entre sí y previniendo la correlación.

El principal problema que presenta Random Forest para este caso es que la mayoría de features son 0 para cada comentario (son 20-50 palabras frente a las posiblemente más de 10.000 del dataset total), lo que hace muy probable que las palabras de un comentario se pierdan en los splits.


## Arquitectura del Código
El proyecto  se compone de dos partes principales: vectorización del texto y entrenamiento/evaluación del modelo.

Todo el proceso se registra en un [notebook]().

### Preprocesado de datos (`vectorize.py`)
Contiene la función que divide el dataset en train/test y se vectorizan los comentarios.

### Vectorización del texto (`model.py`)
Contiene las funciones para entrenar y evaluar los modelos. La función de evaluación imprime la matriz de confusión y el classification report.


## Resultados

|     MODELO           | Macro F1   |   Accuracy |
|        :---          |   :---:    |   :---:    |
|**MultinomialNB**     |    0.63    |    0.75    |
|**ComplementNB**      |    0.64    |    0.76    |
|**LogisticRegression**|    0.65    |    0.74    |             
|**LinearSVC**         |    0.64    |    0.76    |
|**RandomForest**      |    0.50    |    0.73    |


## Modificaciones del Modelo LogisticRegression

|MODELO  LogisticRegression| class_weight  |     C     | Macro F1   |   Accuracy |
|        :---              |   :---:       |   :---:   |   :---:    |   :---:    |
|**Modelo Inicial**        |  *balanced*   | *default* |    0.65    |  0.75      |
|**Modificación 0**        |[2.2, 1.5, 0.5]|   1       |    0.66    |     0.75   |
|**Modificación 1**        |[2.2, 1.5, 1.5]|   1       |    0.63    |    0.77    |
|**Modificación 2**        |[2.2, 1.5, 1.5]|    2.5    |    0.65    |   0.78     |          
|**Modificación 3**        | [2.5, 1.5, 1] |    2.5    |    0.66    |   0.78     |

## Modificaciones del Modelo LinearSVC

|MODELO  LinearSVC         | class_weight  |     C     | Macro F1   |   Accuracy |
|        :---              |   :---:       |   :---:   |   :---:    |   :---:    |
|**Modelo Inicial**        |  *balanced*   | *default* |    0.64    |  0.76      |
|**Modificación 0**        | [3, 1.5, 0.5] |   1       |    0.65    |     0.76   |
|**Modificación 1**        |  [3, 1, 1.5]  |   1       |    0.65    |    0.77    |

## Conclusiones

Los cinco modelos evaluados convergen hacia un techo de macro F1 de aproximadamente 0.65, 
independientemente del algoritmo utilizado o de los ajustes de hiperparámetros realizados. 
La única excepción es Random Forest, que queda significativamente por debajo debido a su 
incompatibilidad con vectores TF-IDF sparse. Este resultado sugiere que la limitación no 
está en los modelos sino en la representación del texto.

El problema principal es la combinación de dos factores. Por un lado el desbalance de clases, 
que arrastra las predicciones hacia la clase mayoritaria y que solo se corrige parcialmente 
mediante `class_weight`. Por otro la naturaleza de TF-IDF como representación bag-of-words: 
trata las palabras como independientes entre sí y sin contexto, lo que dificulta especialmente 
la distinción entre comentarios positivos y neutros, clases semánticamente cercanas que 
comparten gran parte del vocabulario.


## Comparación con Resultados Anteriores

Paradójicamente, el modelo implementado manualmente en el Proyecto 1 obtenía resultados 
comparables gracias a un feature engineering muy específico al dataset: el manejo explícito 
de negaciones con prefijos `NOT_` y un ajuste manual de umbrales. Esto ilustra que un pipeline 
más genérico y profesional no siempre supera al artesanal sin un ajuste fino al dominio, y 
que la clave está en la calidad de la representación del texto.

## Cómo Replicar el Proyecto

* Ejecutar archivo `main.py`. Este devuelve los resultados del entrenamiento y el reporte impresio en pantalla, además de la matriz de confuión, del modelo considerado óptimo. Los parámetros son modificables.

* Ejecutar archivo `notebooks/all_models.ipynb` para ver los resultados de todos los modelos entrenados para el repositorio.







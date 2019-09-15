import numpy as np


class Classifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class ClassifEuclid(Classifier):
    def __init__(self, labels=[]):
        """Constructor de la clase
        labels: lista de etiquetas de esta clase"""
        self.labels = labels
        pass

    def fit(self, x, y):
        """Entrena el clasificador
        X: matriz numpy cada fila es un dato, cada columna una medida
        y: vector de etiquetas, tantos elementos como filas en X
        retorna objeto clasificador"""
        self.predict(x)

        return self

    def predict(self, x):
        """Estima el grado de pertenencia de cada dato a todas las clases
        X: matriz numpy cada fila es un dato, cada columna una medida del vector de caracteristicas.
        Retorna una matriz, con tantas filas como datos y tantas columnas como clases tenga
        el problema, cada fila almacena los valores pertenencia de un dato a cada clase"""
        occurrences = np.unique(self.labels, return_counts=True)
        result = np.zeros((x.shape[0], occurrences[0].size))
        begin = 0
        for i, e in enumerate(occurrences[1]):
            result[:, i] = np.divide(np.sum(x, axis=1), np.mean(np.sum(x[begin:begin+e], axis=1)))
            begin += e

        return result

    def pred_label(self, x):
        """Estima la etiqueta de cada dato. La etiqueta puede ser un entero o bien un string.
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna un vector con las etiquetas de cada dato"""
        result = []
        for e in x:
            result.append((np.abs(e - 1)).argmin())

        return result

    def num_aciertos(self, x, y):
        """Cuenta el numero de aciertos del clasificador para un conjunto de datos X.
        X: matriz de datos a clasificar
        y: vector de etiquetas correctas"""
        same_values = []
        [same_values.append(x[i] == y[i]) for i in range(0, len(labels))]
        num_aciertos = same_values.count(True)
        return num_aciertos, (num_aciertos / len(x)) * 100


if __name__ == "__main__":
    # Load the data
    samples = np.genfromtxt("irisData.txt", usecols=(0, 1, 2, 3), dtype=float)  # 150x4 Matrix
    labels = np.genfromtxt("irisData.txt", usecols=[-1], dtype=str)             # 150x1 Matrix (Vector)

    unique = list(np.unique(labels))
    aux_labels = []

    for index, element in enumerate(labels):
        aux_labels.append(unique.index(element))
    labels = aux_labels
    del aux_labels

    print("Sample matrix dimension:", samples.shape)
    print("Labels matrix dimension:", "(" + str(len(labels)) + ", 1)")

    # print("samples:", samples)
    # print("labels:", labels)

    classifEuclid = ClassifEuclid(labels)

    predict = classifEuclid.predict(samples)
    predict_label = classifEuclid.pred_label(predict)

    num_aciertos = classifEuclid.num_aciertos(predict_label, labels)

    print("Correct answers:", num_aciertos[0], "/", len(labels))
    print("Success rate:", num_aciertos[1])

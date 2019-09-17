import numpy as np
from scipy.spatial import distance


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
        self.coordinates = []
        self.occurrences = []
        pass

    def fit(self, x, y):
        """Entrena el clasificador
        X: matriz numpy cada fila es un dato, cada columna una medida
        y: vector de etiquetas, tantos elementos como filas en X
        retorna objeto clasificador"""
        self.occurrences = np.unique(self.labels, return_counts=True)
        begin = 0
        for e in enumerate(self.occurrences[1]):
            self.coordinates.append((np.mean(x[begin:begin + e[1]], axis=0)))
            begin += e[1]
        """    
        predict_matrix = self.predict(x)
        labels_matrix = self.pred_label(predict_matrix)
        self.num_aciertos(labels_matrix, self.labels)
        """
        return self

    def predict(self, x):
        """Estima el grado de pertenencia de cada dato a todas las clases
        X: matriz numpy cada fila es un dato, cada columna una medida del vector de caracteristicas.
        Retorna una matriz, con tantas filas como datos y tantas columnas como clases tenga
        el problema, cada fila almacena los valores pertenencia de un dato a cada clase"""
        result = np.zeros((x.shape[0], self.occurrences[0].size))

        for j in range(0, len(self.occurrences[1])):
            for i in range(0, len(x)):
                result[i, j] = distance.euclidean(x[i], self.coordinates[j])

        return result

    def pred_label(self, x):
        """Estima la etiqueta de cada dato. La etiqueta puede ser un entero o bien un string.
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna un vector con las etiquetas de cada dato"""
        result = []
        [result.append((e - 1).argmin()) for e in x]

        return result

    def num_aciertos(self, x, y):
        """Cuenta el numero de aciertos del clasificador para un conjunto de datos X.
        X: matriz de datos a clasificar
        y: vector de etiquetas correctas"""
        same_values = []
        [same_values.append(x[i] == y[i]) for i in range(0, len(self.labels))]
        number = same_values.count(True)

        return number, (number / len(x)) * 100


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

    classifEuclid.fit(samples, labels)

    predict_matrix = classifEuclid.predict(samples)

    labels_matrix = classifEuclid.pred_label(predict_matrix)

    correct = classifEuclid.num_aciertos(labels_matrix, labels)

    print("Correct answers:", correct[0], "/", len(labels))
    print("Success rate:", correct[1])
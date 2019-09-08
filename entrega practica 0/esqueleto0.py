import numpy as np


class Classifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class classifEuclid(Classifier):
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
        return self

    def predict(self, x):
        """Estima el grado de pertenencia de cada dato a las clases 
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna una matriz, cada fila almacena los valores pertenencia""" 
        return "..."
    
    def predLabel(self, x):
        """Estima la etiqueta de cada dato
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna un vector con las etiquetas de cada dato"""
        return "..."


# Read the data
samples = np.genfromtxt("irisData.txt", usecols=(0, 1, 2, 3), dtype=float)  # 150x4 Matrix
labels = np.genfromtxt("irisData.txt", usecols=[-1], dtype=str)             # 150x1 Matrix (Vector)

print("Sample matrix dimension", samples.shape)
print("Labels matrix dimension", labels.shape)

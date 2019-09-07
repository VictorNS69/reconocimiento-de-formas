
import numpy as np

class Classifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass

class classifEuclid(Classifier):
    def __init__(self,labels=[]):
        """Constructor de la clase
        labels: lista de etiquetas de esta clase"""
        self.labels=labels
        pass

    def fit(self,X,y):
        """Entrena el clasificador
        X: matriz numpy cada fila es un dato, cada columna una medida
        y: vector de etiquetas, tantos elementos como filas en X
        retorna objeto clasificador"""
        return self

    def predict(self,X):
        """Estima el grado de pertenencia de cada dato a las clases 
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna una matriz, cada fila almacena los valores pertenencia""" 
        return "..."
    
    def predLabel(self,X):
        """Estima la etiqueta de cada dato
        X: matriz numpy cada fila es un dato, cada columna una medida
        retorna un vector con las etiquetas de cada dato"""
        return "..."

# Para leer los datos
samples = np.genfromtxt("irisData.txt", "...")
labels = np.genfromtxt("irisData.txt", "...")

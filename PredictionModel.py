from limpieza_transformer import LimpiezaPreprocesamiento
from joblib import load

class Model:
    def __init__(self,columns):
        self.model = load("modelo_random_forest.pkl")

    def make_predictions(self, data):
        result = self.model.predict(data)
        return result
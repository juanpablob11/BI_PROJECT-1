
from sklearn.base import BaseEstimator, TransformerMixin

class LimpiezaPreprocesamiento(BaseEstimator, TransformerMixin):
    def __init__(self):
        from nltk.corpus import stopwords
        self.stop_words = set(stopwords.words("spanish"))

    def definir_variables(self, df):
        features = ['Titulo', 'Descripcion', 'Label']
        return df[features] if "Label" in df.columns else df[["Titulo", "Descripcion"]]

    def remove_duplicates(self, df):
        return df.drop_duplicates().reset_index(drop=True)

    def eliminar_duplicados_parciales(self, df):
        if "Label" not in df.columns:
            return df
        conteo_labels = df.groupby(['Titulo', 'Descripcion'])['Label'].nunique()
        duplicados_parciales = conteo_labels[conteo_labels > 1].index
        return df[~df.set_index(['Titulo', 'Descripcion']).index.isin(duplicados_parciales)].reset_index(drop=True)

    def limpiar_data(self, df):
        df_variables = self.definir_variables(df)
        df_duplicados = self.remove_duplicates(df_variables)
        return self.eliminar_duplicados_parciales(df_duplicados)

    def preprocessing(self, words):
        import re, unicodedata
        from num2words import num2words

        words = [word.lower() for word in words]
        words = [num2words(word, lang="es") if word.isdigit() else word for word in words]
        words = [re.sub(r'[^\w\s]', '', word) for word in words if word]
        words = [unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore') for word in words]
        words = [word for word in words if word not in self.stop_words]
        return [word for word in words if word.strip() != ""]

    def transform(self, X, y=None):
        import pandas as pd
        import spacy

        df = X.copy()
        df = self.limpiar_data(df)

        df["Titulo"] = df["Titulo"].astype(str).fillna("")
        df["Descripcion"] = df["Descripcion"].astype(str).fillna("")

        nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

        resultados_titulo = [
            [token.text for token in doc if not token.is_space]
            for doc in nlp.pipe(df["Titulo"], batch_size=100)
        ]
        resultados_descripcion = [
            [token.text for token in doc if not token.is_space]
            for doc in nlp.pipe(df["Descripcion"], batch_size=100)
        ]

        df["Titulo_tokens_clean"] = [self.preprocessing(t) for t in resultados_titulo]
        df["Descripcion_tokens_clean"] = [self.preprocessing(d) for d in resultados_descripcion]

        df["Titulo_tokens_clean"] = df["Titulo_tokens_clean"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)
        df["Descripcion_tokens_clean"] = df["Descripcion_tokens_clean"].apply(lambda x: " ".join(x) if isinstance(x, list) else x)

        return df[["Titulo_tokens_clean", "Descripcion_tokens_clean"]]

    def fit(self, X, y=None):
        return self

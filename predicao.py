import joblib

def fazer_previsao(dados):
    modelo = joblib.load("rf_modelo")
    probabilidade = modelo.predict_proba(dados).max()
    classe = modelo.predict(dados)
    resultado = [str(classe[0]),
                 str(round(probabilidade * 100, 1)) + "%"]
    return resultado
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# Controlar aspectos aleatórios da modelagem
seed = 42

# Ler conjunto de dados do arquivo
iris_df = pd.read_csv("dados/iris.csv")

# Selecionar preditores e target
X = iris_df[['comprimento_sepala',
             'largura_sepala',
             'comprimento_petala',
             'largura_petala']]
y = iris_df[['especie']]

# Separar os dados em conjuntos de treino e teste
# 70% para treino e 30% para teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.3, random_state=seed, stratify=y)

# Criar uma floresta aleatória com 100 árvores de decisão
modelo = RandomForestClassifier(n_estimators=100)

# Treinar meu classificador nos dados de treino
# Ou seja, vou ajustar meu modelo
modelo.fit(X_treino, y_treino.values.ravel())

# Fazer previsões nos dados de teste
y_pred = modelo.predict(X_teste)

# Calcular acurácia do modelo
acuracia = accuracy_score(y_teste, y_pred)
print(f"Acurácia: {acuracia}")

# Salvar o modelo em um arquivo local
joblib.dump(modelo, "rf_modelo")
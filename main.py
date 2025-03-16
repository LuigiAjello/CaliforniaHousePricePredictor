import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from scipy.stats import zscore

# Carregar o conjunto de dados
california = fetch_california_housing()
data = pd.DataFrame(california.data, columns=california.feature_names)
data['PRICE'] = california.target

# Remover a primeira linha
first_row = data.iloc[0]  # Armazenar a primeira linha para previsão posterior
data = data.drop(index=0)  # Remover a primeira linha do conjunto de dados

# Função para detectar outliers com base no Z-score
def detect_outliers_zscore(df, threshold=3):
    # Calcular o Z-score para todas as colunas
    z_scores = np.abs(zscore(df))
    # Identificar outliers: valores com Z-score maior que o limiar
    outliers = (z_scores > threshold)
    return outliers

# Detectar outliers no conjunto de dados
outliers = detect_outliers_zscore(data.drop(columns=['PRICE']))

# Remover as linhas com outliers
data_cleaned = data[~outliers.any(axis=1)]

# Separar variáveis independentes (X) e dependente (y)
X_cleaned = data_cleaned.drop(columns=['PRICE'])
y_cleaned = data_cleaned['PRICE']

# Dividir os dados em treino e teste
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Normalizar os dados
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Criar e treinar o modelo de Regressão Linear
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Função para fazer previsões com base em novos dados
def predict_price(input_data):
    # Transformar a entrada para um DataFrame com as mesmas colunas que o modelo espera
    input_data_df = pd.DataFrame([input_data], columns=X_cleaned.columns)
    
    # Normalizar os dados de entrada com o mesmo scaler
    input_data_scaled = scaler.transform(input_data_df)
    
    # Fazer a previsão
    predicted_price = model.predict(input_data_scaled)
    return predicted_price[0]

# Testar a previsão com a primeira linha que foi removida
predicted_price = predict_price(first_row[:-1])  # Excluir o valor de PRICE da primeira linha

print(f"Preço previsto para os dados de entrada (primeira linha removida): ${predicted_price:.2f}")

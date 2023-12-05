import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def descriptive_statistics(df):
    print("Estatísticas Descritivas:", df.describe())
    print("\nValores Ausentes:", df.isnull().sum())

def visualize_data(df, features):
    # Histogramas
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 4, i+1)
        sns.histplot(df[feature], kde=True)
        plt.title(f'Distribuição de {feature}')
    plt.tight_layout()
    plt.show()

    # Mapa de calor da correlação
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.show()

    # Boxplots
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(3, 4, i+1)
        sns.boxplot(y=df[feature])
        plt.title(f'Boxplot de {feature}')
    plt.tight_layout()
    plt.show()

def prepare_data(df, features, target):
    X = df[features]
    y = df[target]

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    y = scaler_y.fit_transform(y.values.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    joblib.dump(scaler_x, 'scaler_x.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')

    return X_train, X_test, y_train, y_test

def revert_scaling(scaler, y):
    y_reverted = scaler.inverse_transform(y)
    return y_reverted

def compare(y_test, predictions):
    plt.figure(figsize=(10, 8))
    # Histograma para os Valores Reais
    sns.histplot(y_test, color='blue', label='Valor Real', kde=True, stat='density', alpha=0.5)
    # Histograma para as Previsões
    sns.histplot(predictions, color='red', label='Previsão', kde=True, stat='density', alpha=0.5)

    plt.title('Comparação entre Valor Real e Previsão')
    plt.legend()
    plt.show()

def plot_residuals(y_test, predictions):
    residuals = y_test - predictions
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.title('Gráfico de Resíduos')
    plt.xlabel('Previsões')
    plt.ylabel('Resíduos')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.show()

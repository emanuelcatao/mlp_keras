from src.data_preparation import compare, plot_residuals, load_data, descriptive_statistics, visualize_data, prepare_data, revert_scaling
from src.model import build_model
from src.train_evaluate import train_evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

def main():
    # Caminho do arquivo de dados
    filepath = './data/dataset.csv'

    # Carregar e preparar os dados
    df = load_data(filepath)
    descriptive_statistics(df)
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'duration_ms']
    #visualize_data(df, features)
    X_train, X_test, y_train, y_test = prepare_data(df, features, 'popularity')

    # Construir e compilar o modelo
    model = build_model(X_train.shape[1])

    # Treinar e avaliar o modelo
    history, predictions_original = train_evaluate(model, X_train, y_train, X_test, y_test, 'scaler_y.pkl')
    scaller = joblib.load('scaler_y.pkl')

    mae = mean_absolute_error(revert_scaling(scaller, y_test), predictions_original)
    mse = mean_squared_error(revert_scaling(scaller, y_test), predictions_original)
    rmse = np.sqrt(mse)

    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)

    scaller = joblib.load('scaler_y.pkl')
    compare(revert_scaling(scaller, y_test), predictions_original)
    plot_residuals(revert_scaling(scaller, y_test), predictions_original)

if __name__ == "__main__":
    main()

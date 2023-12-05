import joblib

def train_evaluate(model, X_train, y_train, X_test, y_test, scaler_y_path):
    # Treinamento do modelo
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
    
    # Avaliação do modelo
    loss = model.evaluate(X_test, y_test)
    print(f"Loss: {loss}")

    # Carregar o scaler_y para inverter as previsões
    scaler_y = joblib.load(scaler_y_path)

    # Fazer previsões com o modelo
    predictions_scaled = model.predict(X_test)

    # Reverter as previsões para a escala original
    predictions_original = scaler_y.inverse_transform(predictions_scaled)
    print("Previsões revertidas para a escala original:", predictions_original)

    return history, predictions_original

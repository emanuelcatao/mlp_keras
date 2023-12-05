from keras.models import load_model
import numpy as np

# Carregar o modelo salvo
model = load_model('C:/Users/ecata/Projects/ia_rossini/mlp_keras/estudo/my_keras_model.h5')

# Solicitar um input do usuário
input_data = []
features = ["Mediana da idade da habitação", "Renda média", "Número médio de cômodos por habitação",
            "Número médio de quartos por habitação", "População", "Número médio de famílias",
            "Latitude", "Longitude"]

for feature in features:
    value = float(input(f"Digite o valor para {feature}: "))
    input_data.append(value)

# Converter os dados de entrada para um array NumPy e remodelar para o formato correto
input_data_np = np.array([input_data])  # Adiciona uma dimensão extra

# Fazer a predição
prediction = model.predict(input_data_np)

print("A predição é:", prediction)

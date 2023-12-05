from data_preprocessing import load_dataset, select_features, scale_features
from clustering import calculate_elbow_silhouette, optimal_number_of_clusters, train_kmeans_model
from recommendation_engine import search_songs, user_song_selection, recommend_songs, open_spotify_track
from visualization import plot_pca, plot_elbow_method, plot_silhouette_coefficient
from data_analysis import show_basic_statistics, plot_feature_distribution, plot_correlation_matrix, plot_pairwise_relationships, analyse_clusters

def main():
    file_path = './data/dataset.csv'

    dataset = load_dataset(file_path)

    sample_size = 1000 
    dataset_sample = dataset.sample(n=sample_size, random_state=0)
    
    show_basic_statistics(dataset_sample)
    features = [
                'danceability', #quão adequada para dançar é uma música
                'energy', #representa uma medida perceptual de intensidade e atividade
                'loudness', #representa a média em decibéis (dB) de uma música
                'speechiness', #detecta a presença de palavras faladas em uma música
                'acousticness', #uma confiança de 0,0 a 1,0 de que a faixa é acústica
                'instrumentalness', #prediz se uma faixa não contém vocais
                'liveness', #detecta a presença de uma audiência na gravação
                'valence', #mede a positividade musical transmitida por uma faixa
                'tempo' #tempo estimado da batida da música
                ]
    plot_feature_distribution(dataset_sample, features)
    plot_correlation_matrix(dataset_sample, features)
    plot_pairwise_relationships(dataset_sample, features)
    
    # Preparação e normalização das características para todo o conjunto de dados
    X = select_features(dataset, features)
    X_scaled, scaler = scale_features(X)

    # Visualização dos dados (PCA)
    plot_pca(X_scaled)

    # Calcular o Método Elbow e o Coeficiente de Silhueta
    k_range = range(2, 21)
    inertia_values, silhouette_scores = calculate_elbow_silhouette(X_scaled, k_range)

    # Encontrar o número ótimo de clusters
    optimal_k = optimal_number_of_clusters(inertia_values)
    print("Número ótimo de clusters:", optimal_k)

    # Plotar Elbow e Silhueta
    plot_elbow_method(k_range, inertia_values)
    plot_silhouette_coefficient(k_range, silhouette_scores)

    # Treinar o modelo K-Means
    kmeans_model = train_kmeans_model(X_scaled, optimal_k)

    # Adicionando a coluna de cluster ao dataset
    dataset['cluster'] = kmeans_model.predict(X_scaled)
    analyse_clusters(kmeans_model, optimal_k, X_scaled, features, scaler)

    song_query = input("Digite o nome da música que você está procurando: ")
    songs_found = search_songs(song_query, dataset)

    if songs_found.empty:
        print("Nenhuma música encontrada com esse nome.")
        return

    selected_song = user_song_selection(songs_found)

    if selected_song is not None:
        recommendations = recommend_songs(selected_song, dataset, features, num_recommendations=5)
        print("Músicas recomendadas:")
        print(recommendations[['track_name', 'artists', 'track_id']])

        # Permitir ao usuário abrir uma música recomendada no Spotify
        track_id_input = input("Digite o ID da música que você quer ouvir no Spotify ou 'sair' para finalizar: ")
        if track_id_input.lower() != 'sair':
            open_spotify_track(track_id_input)
        

if __name__ == "__main__":
    main()

from sklearn.metrics.pairwise import euclidean_distances
import webbrowser

def search_songs(song_name, dataset):
    """Busca músicas que contenham o nome fornecido."""
    return dataset[dataset['track_name'].str.contains(song_name, case=False, na=False)]

def user_song_selection(songs):
    """Permite ao usuário selecionar uma música a partir de uma lista de músicas encontradas."""
    print("Músicas encontradas:")
    for idx, song in songs.iterrows():
        print(f"{idx}: {song['track_name']} by {song['artists']}")

    selected_index = int(input("Insira o número da música que você está buscando: "))
    if selected_index in songs.index:
        return songs.loc[selected_index]
    else:
        print("Não existe essa música, tente novamente.")
        return None

def recommend_songs(selected_song, dataset, features, num_recommendations=5):
    """Recomenda músicas do mesmo cluster da música selecionada com base na menor distância euclidiana."""
    cluster = selected_song['cluster']
    selected_song_name = selected_song['track_name']
    
    # somnte musicas no mesmo cluster
    cluster_songs = dataset[dataset['cluster'] == cluster]
    
    # calculo da distância entre as musicas
    selected_song_features = selected_song[features].values.reshape(1, -1)
    cluster_songs_features = cluster_songs[features]
    distances = euclidean_distances(selected_song_features, cluster_songs_features)[0]
    
    # adicionando ao dataframe e ordenando
    cluster_songs = cluster_songs.assign(distance=distances)
    closest_songs = cluster_songs.sort_values('distance')
    
    # excluindo a música selecionada da lista
    closest_songs = closest_songs[closest_songs['track_name'] != selected_song_name]

    # as menores distancias somente
    return closest_songs.iloc[:num_recommendations]

def open_spotify_track(track_id):
    """Abre a faixa do Spotify no navegador usando o track_id."""
    base_url = "https://open.spotify.com/track/"
    url = base_url + track_id
    webbrowser.open(url)
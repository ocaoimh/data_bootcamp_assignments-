from flask import Flask, render_template, request

import spotipy
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from spotipy.oauth2 import SpotifyClientCredentials
from sklearn.metrics import pairwise_distances_argmin_min


app = Flask(__name__, static_folder='static')

# Initialize SpotiPy with user credentials
client_id = "af3a4e21d9974f798b0ddef081728f2b"
client_secret = "99a65d20eff04d64bcf24b11824dffc4"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)



@app.route('/recommend', methods=['POST'])
def recommend():
    song_name = request.form['song_name']
    recommended_song = recommend_song(song_name)
    return render_template('recommendation.html', song=recommended_song)


# Reading data - these are lists of songs that were popular in the 1980s in Spain and France
songsFrance = pd.read_csv(r'./../data/songsFrance.csv')
songsEspana = pd.read_csv(r'./../data/songsEspana.csv')

# merge the data sets
eighties = [songsEspana, songsFrance]
eighties = pd.concat(eighties)
eighties.columns = [x.lower() for x in eighties.columns]
eighties.columns = eighties.columns.str.replace("[ ]", "_", regex=True)

# drop data we don't need
columns_to_drop = ['unnamed:_0', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'duration_ms', 'time_signature',
                   'album', 'track_id']
df = eighties.drop(columns=columns_to_drop)
df = pd.DataFrame(df)

# Remove duplicates from the DataFrame
df = df.drop_duplicates(subset=['track', 'artist'])

# Reset the index
df = df.reset_index(drop=True)

# Select the features that you need
x = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
        'liveness', 'valence', 'tempo']]

# Standardize the data
scaler = StandardScaler()
x_prep = scaler.fit_transform(x)

# Train and predict
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_prep)
clusters = kmeans.predict(x_prep)

# Create new DataFrame with title, artist, and cluster assigned
scaled_df = pd.DataFrame(x_prep, columns=x.columns)
scaled_df['track'] = df['track']
scaled_df['artist'] = df['artist']
scaled_df['cluster'] = clusters

scaled_df.groupby(['cluster', 'artist'], as_index=False).count().sort_values(['cluster', 'key'],
                                                                            ascending=[True, False])[
    ['artist', 'cluster', 'key']].reset_index(drop=True)


def recommend_song(song_name):
    results = sp.search(q=f'track:{song_name}', limit=1)
    track_id = results['tracks']['items'][0]['id']
    audio_features = sp.audio_features(track_id)
    df_ = pd.DataFrame(audio_features)
    new_features = df_[x.columns]
    scaled_x = scaler.transform(new_features)
    cluster = kmeans.predict(scaled_x)
    filtered_df = scaled_df[scaled_df['cluster'] == cluster[0]][x.columns]
    closest, _ = pairwise_distances_argmin_min(scaled_x, filtered_df)
    print('\n [RECOMMENDED SONG]')
    return ' - '.join([scaled_df.loc[closest]['track'].values[0], scaled_df.loc[closest]['artist'].values[0]])


if __name__ == '__main__':
    app.run(debug=True, port=8000)
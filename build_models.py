import ast
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

base_dir = Path(__file__).resolve().parent
model_dir = base_dir / 'model'
model_dir.mkdir(exist_ok=True)

movies = pd.read_csv(base_dir / 'data' / 'tmdb_5000_movies.csv')
credits = pd.read_csv(base_dir / 'data' / 'tmdb_5000_credits.csv')

movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies = movies.dropna(subset=['overview', 'genres', 'keywords', 'cast', 'crew'])


def convert(text):
    if pd.isna(text):
        return []
    try:
        return [item['name'] for item in ast.literal_eval(text)]
    except (ValueError, SyntaxError):
        return []


def fetch_director(text):
    if pd.isna(text):
        return []
    try:
        return [item['name'] for item in ast.literal_eval(text) if item.get('job') == 'Director']
    except (ValueError, SyntaxError):
        return []


def collapse(items):
    return [item.replace(' ', '') for item in items]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: x[:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])
new['tags'] = new['tags'].apply(lambda x: ' '.join(x))
new = new.dropna(subset=['tags'])

cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()
similarity = cosine_similarity(vector)

with open(model_dir / 'movie_list.pkl', 'wb') as f:
    pickle.dump(new, f)

with open(model_dir / 'similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print(f'Created movie list with {len(new)} movies')
print(f'Saved to {model_dir / "movie_list.pkl"}')
print(f'Saved to {model_dir / "similarity.pkl"}')

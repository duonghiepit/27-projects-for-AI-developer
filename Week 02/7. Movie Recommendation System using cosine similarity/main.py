# Import Libraries
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

data = {
    'movie_id': [1, 2, 3, 4, 5],
    'title': ['The Matrix', 'John Wick', 'The Godfather', 'Pubp Fiction', 'Cô dâu 8 tuổi'],
    'genre': ['Action, Sci-Fi', 'Action, Thriller', 'Crime, Drama', 'Crime, Drama', 'Action, Crime, Drama']
}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

print("Movie data:")
print(df)

# Define a TF-IDF Vectorizer to transform the genre text into vectors
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the genre column into a matrix of TF-IDF features
tfidf_matrix = tfidf.fit_transform(df['genre'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)

# Funtion to recommend movies based on cosine similarity
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movies that matches the title
    idx = df[df['title'] == title].index[0]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the indices of the 2 most similar movies
    sim_scores = sim_scores[:2]
    
    # Get the movie indexes
    movie_indices = [i[0] for i in sim_scores]

    return df['title'].iloc[movie_indices]

# Test the recommendation system
movie_title = 'The Matrix'
recommended_movies = get_recommendations(movie_title)

print(f"Movie recommended for '{movie_title}':")
for movie in recommended_movies:
    print(movie)
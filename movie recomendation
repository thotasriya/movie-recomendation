import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

# Load the datasets
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')
links_df = pd.read_csv('links.csv')
tags_df = pd.read_csv('tags.csv')

# Merge datasets to get the movie titles along with ratings
ratings_movies_df = ratings_df.merge(movies_df, on='movieId')

# Handle duplicate ratings (by averaging them)
ratings_movies_df = ratings_movies_df.groupby(['userId', 'title']).agg({'rating': 'mean'}).reset_index()

# Create a user-item matrix
user_movie_ratings = ratings_movies_df.pivot(index='userId', columns='title', values='rating').fillna(0)

# Normalize the ratings (subtract mean rating for each user)
user_ratings_mean = np.mean(user_movie_ratings, axis=1)
ratings_demeaned = user_movie_ratings.values - user_ratings_mean.values.reshape(-1, 1)

# Perform Singular Value Decomposition (SVD)
U, sigma, Vt = svds(ratings_demeaned, k=50)

# Convert sigma to diagonal matrix form
sigma = np.diag(sigma)

# Calculate predicted ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_movie_ratings.columns)

def recommend_movies(user_id, predicted_ratings_df, movies_df, ratings_df, num_recommendations=5):
    # Get and sort the user's predicted ratings
    user_row_number = user_id - 1  # UserID starts from 1, index starts from 0
    sorted_user_predictions = predicted_ratings_df.iloc[user_row_number].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information
    user_data = ratings_df[ratings_df.userId == user_id]
    user_full = user_data.merge(movies_df, on='movieId').sort_values(['rating'], ascending=False)
    
    print(f'User {user_id} has already rated {user_full.shape[0]} movies.')
    print(f'Recommendations for user {user_id}:')

    # Recommend the highest predicted rating movies that the user hasn't seen yet
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
                        .merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='title')
                        .rename(columns={user_row_number: 'PredictedRating'})
                        .sort_values('PredictedRating', ascending=False)
                        .iloc[:num_recommendations, :-1])
    
    return user_full, recommendations

# Prompt user for ID and display recommendations
user_id = int(input("Enter user ID for which you want recommendations: "))
already_rated, predictions = recommend_movies(user_id, predicted_ratings_df, movies_df, ratings_df)

print("\nAlready rated movies:")
print(already_rated[['title', 'genres', 'rating']])

print("\nRecommended movies:")
print(predictions[['title', 'genres']])
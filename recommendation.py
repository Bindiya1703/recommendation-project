import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import numpy as np
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
data = pd.merge(ratings, movies, on='movieId')
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix = user_item_matrix.fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
def recommend_movies(user_id, num_recommendations=5):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).index[1:]
    recommendations = []
    for similar_user in similar_users:
        similar_user_ratings = user_item_matrix.loc[similar_user]
        similar_user_unrated = similar_user_ratings[similar_user_ratings == 0]
        recommendations.extend(similar_user_unrated.index)
        if len(recommendations) >= num_recommendations:
            break
    return list(set(recommendations))[:num_recommendations]
user_id = 1
recommended_movies = recommend_movies(user_id, 5)
print(f"Movies recommended for user {user_id}: {recommended_movies}")
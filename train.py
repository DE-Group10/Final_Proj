from flask import Flask, request, Response, json
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.simplefilter('ignore')

# load data
df = pd.read_csv("ratings_Electronics.csv", header=None)

# drop campaign related columns
columns = ['userId', 'productId', 'ratings', 'timestamp']
electronics_df = pd.read_csv('ratings_Electronics.csv', names=columns)
electronics_df.drop('timestamp', axis=1, inplace=True)

# Taking subset of the dataset
electronics_df1 = electronics_df.iloc[:50000, 0:]

# Summary statistics of rating variable
electronics_df1['ratings'].describe().transpose()

most_rated = electronics_df1.groupby('userId').size().sort_values(ascending=False)[:10]

counts = electronics_df1.userId.value_counts()
electronics_df1_final = electronics_df1[electronics_df1.userId.isin(counts[counts >= 15].index)]

# Constructing the pivot table
final_ratings_matrix = electronics_df1_final.pivot(index='userId', columns='productId', values='ratings').fillna(0)

# Calculating the density of the rating matrix
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings / possible_num_of_ratings)
density *= 100
print('density: {:4.2f}%'.format(density))

train_data, test_data = train_test_split(electronics_df1_final, test_size=0.3, random_state=0)

# Count of user_id for each unique product as recommendation score
train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
train_data_grouped.rename(columns={'userId': 'score'}, inplace=True)
train_data_grouped.head(40)

# Sort the products on recommendation score
train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending=[0, 1])

# Generate a recommendation rank based upon score
train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first')

# Get the top 5 recommendations
popularity_recommendations = train_data_sort.head(5)


def recommend(user_id):
    user_recommendations = popularity_recommendations

    # Add user_id column for which the recommendations are being generated
    user_recommendations['userId'] = user_id

    # Bring user_id column to the front
    cols = user_recommendations.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    user_recommendations = user_recommendations[cols]

    return user_recommendations


electronics_df_CF = pd.concat([train_data, test_data]).reset_index()
pivot_df = electronics_df_CF.pivot(index='userId', columns='productId', values='ratings').fillna(0)
pivot_df['user_index'] = np.arange(0, pivot_df.shape[0], 1)
pivot_df.set_index(['user_index'], inplace=True)

U, sigma, Vt = svds(pivot_df, k=10)

# Construct diagonal array in SVD
sigma = np.diag(sigma)

# Predicted ratings
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
# Convert predicted ratings to dataframe
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=pivot_df.columns)

# Recommend the items with the highest predicted ratings

def recommend_items(userID, num_recommendations):
    # index starts at 0
    user_idx = userID - 1
    # Get and sort the user's ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    temp = pd.DataFrame(sorted_user_predictions)
    temp.reset_index(level=0, inplace=True)
    temp.columns = ['productId', 'user_predictions']
    temp = temp.head(num_recommendations)
    return temp


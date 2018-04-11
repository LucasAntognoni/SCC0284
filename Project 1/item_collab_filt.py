import numpy as np
import pandas as pd
import math


# Global Variables
NMOVIES = 3564
USERS = []
SIMILARITY = np.zeros((NMOVIES,NMOVIES), dtype=float)
PREDICTIONS = np.zeros((NMOVIES,1), dtype=float)
MOVIE_IDS = []

class User():    
    
    id = 0
    mean = 0
    movies = 0

    def __init__(self, id, mean, movies):
        self.id = id
        self.mean = mean
        self.movies = movies
        
def read_data():
    return pd.read_csv('train_data.csv')

def process_data(data):

    global USERS, MOVIE_IDS
    
    user_ids = data['user_id'].unique().tolist()
    MOVIE_IDS = sorted(data['movie_id'].unique().tolist())

    for id in user_ids:
        movies = data.loc[(data['user_id'] == id), ['movie_id', 'rating']].sort_values('movie_id').as_matrix()
        mean = movies.mean(axis=0)[1]
        USERS.append(User(id, mean, movies))
    
def search(movie_i, movie_j):
    
    sum_a = 0
    sum_b = 0
    sum_c = 0

    for user in USERS:

        if ((np.where(user.movies == movie_i)[0].size != 0) and (np.where(user.movies == movie_j)[0].size != 0 )):
            
            rating_movie_i = user.movies[int((np.where(user.movies == movie_i)[0])[0])][1]
            rating_movie_j = user.movies[int((np.where(user.movies == movie_j)[0])[0])][1]
            
            sum_a += (rating_movie_i - user.mean) * (rating_movie_j - user.mean)
            sum_b += pow((rating_movie_i - user.mean), 2)
            sum_c += pow((rating_movie_j - user.mean), 2)

    if (sum_a == 0 or sum_b == 0 or sum_c == 0):
            return 1, 1, 1
        
    return sum_a, sum_b, sum_c
    
def similarity():
    
    global SIMILARITY

    for movie_i in MOVIE_IDS:
        for movie_j in MOVIE_IDS:
            if movie_i != movie_j:
                sum_i_j, sum_i, sum_j = search(movie_i, movie_j) 
                SIMILARITY[movie_i][movie_j] = sum_i_j / (math.sqrt(sum_i) * math.sqrt(sum_j))
                # print(SIMILARITY[movie_i][movie_j])

def prediction(user, item, numberOfNeighbors):
    pass

def main():
    
    process_data(read_data())
    similarity()

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import math
from timeit import default_timer as timer

def read_data():

    train = pd.read_csv('./data/train_data.csv')
    test = pd.read_csv('./data/test_data.csv')

    return train, test

def naiveBayes(q_id, user, movie, data):
    
    attributes = data.loc[data['user_id'] == user, 'movie_id':'rating']

    # Hypothesis probability
    p_of_v = np.zeros((5))
    
    unique, counts = np.unique(data.loc[data['movie_id'] == movie].rating.values, return_counts=True)
    movie_ratings_frequence = dict(zip(unique, counts))
    
    for i in range(5):
        if i + 1 in movie_ratings_frequence:
            p_of_v[i] = (movie_ratings_frequence[i + 1] / (sum(movie_ratings_frequence.values()))) 
        else:
            p_of_v[i] = 0

    # Conditional probability P(ai|vj)
    p_a_v = np.zeros((len(attributes.index), 5))

    index = 0

    for row in attributes.itertuples():

        for col in range(5):

            a = len(data.loc[(data['movie_id'] == row.movie_id) & (data['rating'] == col + 1)])
            b = len(data.loc[(data['movie_id'] == movie) & (data['rating'] == col + 1)])
            
            if a == 0 or b == 0:
                p_a_v[index][col] = 0
            else:
                p_a_v[index][col] = b / a

        index += 1

    p = np.zeros((5))

    for i in range(5):
        p[i] = p_of_v[1] * np.prod(p_a_v[:,i])

    # print(q_id,np.argmax(p) + 1)

def main():

    data, test = read_data()
    
    start_global = timer()

    for row in test.itertuples():
        start_it = timer()

        naiveBayes(row.id, row.user_id, row.movie_id, data)
        
        end_it = timer()
        time_elapsed_it = end_it - start_it
        print(row.id, time_elapsed_it)

    end_global = timer()
    time_elapsed_global = end_global - start_global
    print(time_elapsed_global)

if __name__ == '__main__':
    main()
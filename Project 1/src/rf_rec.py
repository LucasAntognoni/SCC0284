import numpy as np
import pandas as pd
import math

def read_data():
        
    return pd.read_csv('./data/train_data.csv'), pd.read_csv('./data/test_data.csv')

def rf_rec(id, user, item, data):

    unique, counts = np.unique(data.loc[data['user_id'] == user].rating.values, return_counts=True)
    user_ratings_frequence = dict(zip(unique, counts))

    unique, counts = np.unique(data.loc[data['movie_id'] == item].rating.values, return_counts=True)
    item_ratings_frequence = dict(zip(unique, counts))

    ratings = []

    for i in range(0,6):
        
        if i in user_ratings_frequence:
            a = (user_ratings_frequence[i] + 1)
        else:
            a = 1
            
        if i in item_ratings_frequence:
            b = (item_ratings_frequence[i] + 1)
        else:
            b = 1

        ratings.append(a * b)

    print('%d,%d' % (id, ratings.index(max(ratings))))

def main():
    data, test = read_data()

    print('id,rating')
    for row in test.itertuples():
        rf_rec(row.id, row.user_id, row.movie_id, data)

if __name__ == '__main__':
    main()
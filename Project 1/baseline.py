import numpy as np
import pandas as pd
import math

def read_data():

    train = pd.read_csv('./data/train_data.csv')
    test = pd.read_csv('./data/test_data.csv')

    users = train.user_id.unique()
    mean = np.zeros((users[-1]))

    for user in users:
        mean[user - 1] = train.loc[train['user_id'] == user]["rating"].mean()

    return train, test, mean

def baseline(data, users_mean, global_mean):

    users_bias = np.zeros((len(users_mean)))
    items_bias = np.zeros((np.sort(data.movie_id.unique())[-1]))
    
    users = data.user_id.unique()
    movies = np.sort(data.movie_id.unique())

    for movie in movies:
        diff = np.subtract((data.loc[data['movie_id'] == movie].rating.values), global_mean)
        sum_movie = np.sum(diff)
        movies_bias[movie -1] = sum_movie / len(sum_movie)
    

    for user in users:
        
        diff = np.subtract((data.loc[data['user_id'] == user].rating.values), global_mean)
        
        user_movie_list = data.loc[data['user_id'] == user].movies_id.values
        

        
            
        sum_user = np.sum(diff)
        users_bias[user -1] = sum_user / len(sum_user)
    
    for movi


def main():

    data, test, mean = read_data()
    
    print('id,rating')

    global_users_mean = mean.mean()

    # for row in test.itertuples():
    #     baseline(row.id, row.user_id, row.movie_id, data, mean, global_users_mean)
    baseline(data, mean, global_users_mean)

if __name__ == '__main__':
    main()
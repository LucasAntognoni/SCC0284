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

    return train, test, mean.mean()

def baseline(data, global_mean):

    users = data.user_id.unique()
    movies = np.sort(data.movie_id.unique())
    
    users_bias = np.zeros((users[-1]))
    movies_bias = np.zeros((movies[-1]))
    
    for movie in movies:
        diff = np.subtract((data.loc[data['movie_id'] == movie].rating.values), global_mean)
        sum_movie = np.sum(diff)
        movies_bias[movie - 1] = sum_movie / len(diff)
        # print("Movie[%d] Bias: %f" % (movie, movies_bias[movie - 1]))
    
    for user in users:
        
        diff = np.subtract((data.loc[data['user_id'] == user].rating.values), global_mean)
        user_movie_list = np.subtract((data.loc[data['user_id'] == user].movie_id.values), 1)
        user_movie_bias = np.take(movies_bias, user_movie_list)
        diff = diff + user_movie_bias
        sum_user = np.sum(diff)
        users_bias[user - 1] = sum_user / len(diff)
        # print("User[%d] Bias: %f" % (user, users_bias[user - 1]))

    return movies_bias, users_bias

def predict(q_id, user, movie, m_bias, u_bias, mean):
    
    prediction = mean + u_bias[user - 1] + m_bias[movie - 1]

    print('%d,%f' % (q_id,prediction))

def main():

    data, test, mean = read_data()
    
    movies_bias, users_bias = baseline(data, mean)
    
    for row in test.itertuples():
        predict(row.id, row.user_id, row.movie_id, movies_bias, users_bias, mean)

if __name__ == '__main__':
    main()
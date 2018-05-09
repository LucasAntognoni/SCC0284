import numpy as np
import pandas as pd
import math as mt
from timeit import default_timer as timer

def read_data():

    train = pd.read_csv('./data/train_data.csv')
    test = pd.read_csv('./data/test_data.csv')

    sum_ratings = 0

    ratings = train['rating']

    for r in ratings:
        sum_ratings += r

    mean = sum_ratings / train.shape[0]
    
    return train, test, mean
 

def calculateSVD(data, mean, k, l, r, iterations):
    
    users = data.user_id.unique()[-1]
    movies = np.sort(data.movie_id.unique())[-1]

    b_u = np.zeros((users))
    b_i = np.zeros((movies))
    
    P = np.random.uniform(0,0.1,[users, k])
    Q = np.random.uniform(0,0.1,[movies, k])
    
    error = []

    for i in range(iterations):
        sq_error = 0
        for row in data.itertuples():
            
            u = row.user_id
            i = row.movie_id
            r_u_i = row.rating

            pred = mean + b_u[u - 1] + b_i[i - 1] + np.dot(P[u-1,:], Q[i-1,:])
            e_u_i = r_u_i - pred
            sq_error = r_u_i + (e_u_i * e_u_i)

            b_u[u - 1] = b_u[u - 1] + l * e_u_i
            b_i[i - 1] = b_i[i - 1] + l * e_u_i

            for f in range(k):
                temp_u_f = P[u - 1][f - 1]
                P[u - 1][f - 1] = P[u - 1][f - 1] + l * (e_u_i * Q[i - 1][f - 1] - r * P[u - 1][f - 1])
                Q[i - 1][f - 1] = Q[i - 1][f - 1] + l * (e_u_i * temp_u_f - r * Q[i - 1][f - 1])

        error.append(mt.sqrt(sq_error / len(data.index)))
        
    return b_u, b_i, P, Q, error

def predict(q_id, user, movie, bu, bi, p, q, k, mean):
    
    b_u_i = mean + bu[user - 1] + bi[movie - 1]
    
    s_p_q =  0
    
    for i in range(k):
        s_p_q = s_p_q + (p[user - 1][i] * q[movie - 1][i])
    
    prediction = b_u_i + s_p_q

    # print("%d,%f"  % (q_id,prediction))

def main():

    data, test, mean= read_data()

    start_global = timer()

    bu, bi, p, q, errors = calculateSVD(data, mean, 2, 0.05, 0.002, 10)

    start_global = timer()

    for row in test.itertuples():
        start_it = timer()

        predict(row.id, row.user_id, row.movie_id, bu, bi, p, q, 2, mean)
        
        end_it = timer()
        time_elapsed_it = end_it - start_it
        print(row.id, time_elapsed_it)

    end_global = timer()
    time_elapsed_global = end_global - start_global
    print(time_elapsed_global)

if __name__ == '__main__':
    main()
import numpy as np
import pandas as pd
import math

def read_data():
        
    df = pd.read_csv('train_data.csv')

    users = df.user_id.unique()
    mean = np.zeros((users[-1]))

    for user in users:
        mean[user - 1] = df.loc[df['user_id'] == user]["rating"].mean()

    return df, mean

def similarity(data, mean):

    movies = np.sort(data.movie_id.unique())
    similarity_matrix = np.zeros((movies[-1], movies[-1]))

    # Movies size 3562
    # Last movie_id 3564

    for mi in movies:
        for mj in movies:   
            
            if mi != mj:
                
                query = data.loc[(data['movie_id'] == mi) | (data['movie_id'] == mj)]
                query = query[query.groupby('user_id').user_id.transform(len)>1]
            
                user_list = query.user_id.unique()
                
                sum_a = 0
                sum_b = 0
                sum_c = 0
                
                for user in user_list:
                
                    diff = np.subtract((query.loc[query['user_id'] == user].rating.values), mean[user-1])

                    sum_a += np.prod(diff)
                    sum_b += diff[0] ** 2
                    sum_c += diff[1] ** 2

                if (sum_b == 0 or sum_c == 0):
                    similarity_matrix[mi - 1][mj - 1] = 0
                else:
                    similarity_matrix[mi - 1][mj - 1] = sum_a / (math.sqrt(sum_b) * math.sqrt(sum_c))
                
                print('Similarity[%d][%d]: %f' % (mi, mj, similarity_matrix[mi - 1][mj - 1]))
            
    return similarity_matrix

def prediction(user, item, neighbors, similarity, data):
    
    user = data.loc[data['user_id'] == user]
    similar_itens = similarity[item - 1][:]
    sorted_indexes = np.argsort(similar_itens)[0:neighbors]

    sum_a = 0
    sum_b = 0

    for index in sorted_indexes:
        sum_a += similar_itens[index] * user.loc[user['movie_id'] == (index + 1)].rating.values[0]
        sum_b += similar_itens[index]

    print(sum_a / sum_b)


def main():
    data, mean = read_data()
    matrix = similarity(data, mean)
    # prediction(1, 1, 4, matrix, data)

if __name__ == '__main__':
    main()
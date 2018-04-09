import numpy as np
import pandas as pd

class User():    
    
    id = 0
    mean = 0
    movies = np.zeros((1,2), dtype=np.uint16)

    def __init__(self, id):
        self.id = id
        
def read_data():
    return pd.read_csv('train_data.csv')

def process_data(data):

    number_of_users = data['user_id'].max()
    users = []

    for id in range(number_of_users):
        users.append(User(id + 1))

    for user in users:
        
        movies_list = data.loc[data['user_id'] == user.id]
        movies_list = movies_list.sort_values('movie_id')

        user.mean = data["rating"].mean()

        for index, movie in movies_list.iterrows():
            
            user.movies = np.append(user.movies, [[movie[1], movie[2]]], axis=0)
        
        user.movies = np.delete(user.movies, 0, 0)     
    
    return users

def similarity(firstItem, secondItem):
    pass

def prediction(user, item, numberOfNeighbors):
    pass

def main():
    data = read_data()
    u = process_data(data)

if __name__ == '__main__':
    main()
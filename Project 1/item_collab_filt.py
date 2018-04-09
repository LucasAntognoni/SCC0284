import numpy as np
import pandas as pd

class User():    
    
    id = 0
    mean = 0
    movies = pd.DataFrame()

    def __init__(self, id, mean, movies):
        self.id = id
        self.mean = mean
        self.movies = movies
        
def read_data():
    return pd.read_csv('train_data.csv')

def process_data(data):

    number_of_users = data['user_id'].max()
    users = []

    for id in range(number_of_users):
    
        movies = data.loc[(data['user_id'] == id + 1), ['movie_id', 'rating']].sort_values('movie_id')
        mean = movies["rating"].mean()
        users.append(User(id + 1, mean, movies))

    # print("ID: ", users[0].id)
    # print("Mean: ", users[0].mean)
    # print("Movies: ", users[0].movies)

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
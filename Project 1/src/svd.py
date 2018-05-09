import numpy as np
import math as mt
import pandas as pd
import scipy as sp
import sparsesvd as svd
from timeit import default_timer as timer

def read_data():

    train = pd.read_csv('./data/train_data.csv')
    test = pd.read_csv('./data/test_data.csv')

    data = np.zeros((train.user_id.unique()[-1], np.sort(train.movie_id.unique())[-1]))

    for row in train.itertuples():
        data[row.user_id -1][row.movie_id -1] = row.rating

    return data, test

def computeSVD(data, K):
	
    smat = sp.sparse.csc_matrix(data)

    P, s, Q = svd.sparsesvd(smat, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])
            
    P = sp.sparse.csc_matrix(np.transpose(P), dtype=np.float32)
    S = sp.sparse.csc_matrix(S, dtype=np.float32)
    Q = sp.sparse.csc_matrix(Q, dtype=np.float32)
    
    return P.toarray(), S.toarray(), Q.toarray()

def predict(q_id, user, movie, P, S, Q, K):
	
    prediction = 0

    for i in range(K):
        prediction = prediction + (P[user - 1][i] * S[i] * Q[i][movie - 1])

    # print("%d,%f"  % (q_id,np.argmax(prediction) + 1))

def main():
    
    data, test = read_data()
    
    start_global = timer()

    P, S, Q = computeSVD(data, 2)

    for row in test.itertuples():
        start_it = timer()
        
        predict(row.id, row.user_id, row.movie_id, P, S, Q, 2)

        end_it = timer()
        time_elapsed_it = end_it - start_it
        print(row.id, time_elapsed_it)

    end_global = timer()
    time_elapsed_global = end_global - start_global
    print(time_elapsed_global)
    

if __name__ == '__main__':
    main()
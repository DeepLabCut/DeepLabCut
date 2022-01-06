import numpy as np
import pickle
# fix the random seed
np.random.seed(0)


def split_npy(fname, animal):
    vectors = np.load(fname, allow_pickle=True)

    n_samples = vectors.shape[0]

    indices = np.random.permutation(n_samples)

    num_train = int(n_samples * 0.8)

    vectors = vectors[indices]


    train = vectors[:num_train]

    test = vectors[num_train:]

    print ('train shape')
    print (train.shape)

    with open(f'{animal}_train.npy', 'wb') as f:
        np.save(f, train)

    with open(f'{animal}_test.npy', 'wb') as f:
        np.save(f, test)    


def split_pickle(fname, animal, train_test_split = False):
    with open(fname, 'rb') as f:
        a = pickle.load(f)
    
        vectors = a['vectors']
        gts = a['gts']
    
    if train_test_split:
                
        n_samples = vectors.shape[0]

        indices = np.random.permutation(n_samples)

        num_train = int(n_samples * 0.8)

        vectors = vectors[indices]

        gts = gts[indices]

        x_train = vectors[:num_train]
        y_train = gts[:num_train]

        x_test = vectors[num_train:]
        y_test = gts[num_train:]


        train_dict = {'vectors':x_train, 'gts': y_train}
        test_dict = {'vectors':x_test, 'gts':y_test}


        with open(f'{animal}_pair_train.pickle', 'wb') as f:
            pickle.dump(train_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        test_dict = {'vectors':vectors, 'gts': gts}
        
    with open(f'{animal}_pair_test.pickle', 'wb') as f:
        pickle.dump(test_dict, f, protocol=pickle.HIGHEST_PROTOCOL)        

if __name__ == '__main__':
    import sys
    animal = sys.argv[1]
    split_npy(f'{animal}_hard_triplet_vectors.npy', animal)
    #split_pickle(f'{animal}_pair_dataset.pickle', animal)

"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""
import os
import pickle
import pandas as pd


def attempttomakefolder(foldername):
    if os.path.isdir(foldername):
        print("Folder already exists!")
    else:
        os.mkdir(foldername)


def SaveData(PredicteData, metadata, dataname, pdindex, imagenames):
    DataMachine = pd.DataFrame(PredicteData, columns=pdindex, index=imagenames)
    DataMachine.to_hdf(dataname, 'df_with_missing', format='table', mode='w')
    with open(dataname.split('.')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)


def get_immediate_subdirectories(a_dir):
    # https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    return [
        name for name in os.listdir(a_dir)
        if os.path.isdir(os.path.join(a_dir, name))
    ]

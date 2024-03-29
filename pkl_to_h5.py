import os.path as osp
import os 
import pickle
import pandas as pd 

''' conver pkl to h5 file for visualize (vis. by vitables)
'''

if __name__ == '__main__':
    # path = '.\data\\training_pickle\\'
    # unpickled_df = pd.read_pickle(osp.join(path, "Training_Data_Agmented.pkl"))
    
    unpickled_df = pd.read_pickle('NewDetectCache.pkl')
    df = pd.DataFrame(unpickled_df)

    df.info()
    df.to_hdf('NewDetectCache.h5', key='df', mode='w')


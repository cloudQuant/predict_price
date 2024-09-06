import pandas as pd
import numpy as np
import pickle


def clean_origin_data(file_name):
    data = pd.read_csv(f'/home/yun/Downloads/{file_name}.csv')
    print(data.columns)
    print(data.head())
    data = data[['datetime', 'o', 'h', 'l', 'c', 'vol']]
    data.columns = ['datetime','open','high','low','close','volume']
    # with open(f"{file_name}.pkl", 'wb') as f:
    #     pickle.dump(data, f)
    data.to_pickle(f'{file_name}.pkl.gz', compression='gzip')


if __name__ == '__main__':
    clean_origin_data("BTC")
    clean_origin_data("ETH")
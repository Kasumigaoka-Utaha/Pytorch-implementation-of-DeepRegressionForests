import os
import csv
import glob
import re
import pandas as pd

def get_info(path):
    info_list = []
    column_name = ['filename','age']
    for file in glob.glob(path + '*.JPG'):
        name = os.path.basename(file)
        filename = name.split('.')[0]
        num_list = [int(s) for s in re.findall(r'\d+', filename)]
        age = num_list[1]
        value = (filename,age)
        info_list.append(value)
    info_df = pd.DataFrame(info_list,column_name)
    return info_df

if __name__ == "__main__":
    path = './FGNET/images/'
    info_df = get_info(path)
    info_df.to_csv('info.csv',index = None)
    print('Finished data preprocessing...')


    




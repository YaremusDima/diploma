import razdel
import pandas as pd
import os
from sklearn.model_selection import train_test_split

dir_path = 'datasets'
new_dir_path = 'datasets_prepared'

for file in os.listdir(dir_path):
    dataset = pd.read_csv(os.path.join(dir_path, file), delimiter='\t')
    dataset_short = dataset if len(dataset) <= 500000 else dataset.sample(500000)
    dataset_short.replace('þ<br />þ', '\n ', regex=True, inplace=True)
    dataset_train, dataset_test = train_test_split(dataset_short, test_size=0.05, shuffle=True)
    train_file_name = os.path.join(new_dir_path, file[:-4] + '_train.csv')
    test_file_name = os.path.join(new_dir_path, file[:-4] + '_test.csv')
    print(f'train size {len(dataset_train)}, train file name {train_file_name}')
    print(f'test size {len(dataset_test)}, test file name {test_file_name}')
    dataset_train.to_csv(train_file_name, index=False)
    dataset_test.to_csv(test_file_name, index=False)

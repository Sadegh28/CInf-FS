import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
from CInfFS import CInfFS
from classify import classify


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, required=False, default='data')
    parser.add_argument('--datasets', type=str, nargs='+', required=True)
    parser.add_argument('--output-path', type=str, required='False', default='results')
    parser.add_argument('--fs-methods', type=str, nargs='+', required=False, default='CIFS')
    parser.add_argument('--num-of-features', type=int, required=False, default=50)
    parser.add_argument('--classifiers', type=str, nargs='+', required=False, default=None)
    parser.add_argument('--metrics', type=str, nargs='+', required=False, default='None')
    parser.add_argument('--num-of-clusters', type=int, required=False, default=2)
    parser.add_argument('--coeff', type=float, required=False, default=.8)

    args = parser.parse_args() 

    for d in args.datasets: 
        if (not os.path.isfile(args.data_path  + '\\' + d + '.csv')): 
            raise ValueError('The dataset {} is not found :('.format(d))
        data_file = args.data_path  + '\\' + d + '.csv'
        data = pd.read_csv(data_file)
        data = data.to_numpy()
        
        X = data[:,:-1]
        y = data[:,-1]

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.3, train_size =0.7, random_state=42)

        for method in args.fs_methods: 
            start = time.time()
            message = 30*'='+'  dataset:{}  method:{}'.format(d,method)+30*'='
            print(message)
            print(method)
            if method == 'IFS':                 
                fs = CInfFS(num_clusters=1)
            if method == 'CIFS': 
                fs = CInfFS(num_clusters=args.num_of_clusters, merge_coeff= args.coeff)
            
            print('Ranking Features ...')
            rank = fs.rank(x_train,y_train)
            end = time.time()

            # writing the selected subsets into file
            dir_name = args.output_path + '\SelectedSubsets' 
            if not (os.path.isdir(dir_name)):
                os.mkdir(dir_name) 
            dir_name += r'\{}'.format(d)
            if not (os.path.isdir(dir_name)):
                os.mkdir(dir_name)
            filename = dir_name + r'\\' + method + '.csv'
            np.savetxt(filename, rank, delimiter=',', fmt = '%d')

            # writing the running time into file
            dir_name = args.output_path + r'\RunningTimes' 
            if not (os.path.isdir(dir_name)):
                os.mkdir(dir_name)
            dir_name += r'\{}'.format(d)
            if not (os.path.isdir(dir_name)):
                os.mkdir(dir_name) 
            filename = dir_name + r'\\' + method +  '.txt'
            np.savetxt(filename, [end-start], fmt = '%d')

            if args.classifiers != None:
                 

                for c in args.classifiers:
                    dir_name = args.output_path + '\\' + "Accuracies" 
                    if not (os.path.isdir(dir_name)):
                        os.mkdir(dir_name)
                    dir_name += r'\{}'.format(d) 
                    if not (os.path.isdir(dir_name)):
                        os.mkdir(dir_name) 
                    dir_name += r'\{}'.format(c)
                    if not (os.path.isdir(dir_name)):
                        os.mkdir(dir_name) 
                    with tqdm(total=args.num_of_features, ncols=80) as t:
                        t.set_description('{} Classification in Progress '.format(c))
                        for k in range(1, args.num_of_features+1):
                            res = classify(x_train[:,rank[:k]], y_train, x_test[:,rank[:k]], y_test, c, args.metrics)
                            
                            for m in args.metrics: 
                                dir_name_m = dir_name +r'\{}'.format(m)
                                if not (os.path.isdir(dir_name_m)):
                                    os.mkdir(dir_name_m) 
                                filename = dir_name_m +  "\\" +  method +  '.csv'
                                if k == 1:
                                    np.savetxt(filename, [res[m]])
                                else: 
                                    with open(filename, "ab") as f:
                                        np.savetxt(f, [res[m]])


                            t.update(1)


        
        
        
        







        
            
            
            







        
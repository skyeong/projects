import argparse
import os
import numpy as np
import pandas as pd
from kakaobank import checkOutputFolder
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def kmeans_performance(X,kval):
    # Inertia: Sum of distances of samples to their closest cluster center
    sse = dict()
    for k in range(kval):
        models = KMeans(n_clusters=k+1, random_state=0).fit(X)
        sse[k] = models.inertia_
    return sse


def kmeans_analysis(X,kval):
    models = KMeans(n_clusters=kval, random_state=0).fit(X)
    results = dict()
    for label in range(kval):
        idx = np.where(models.labels_==label)[0]
        mu=[np.mean(X[idx,i]) for i in range(X.shape[1])]
        N = len(np.where(models.labels_==label)[0])
        results[label] = {'mean':mu, 'N':N,'idx':idx}
    
    return results


def analysis_k_vs_sse(proj_path,colnames,kmax):

    for year in range(2008,2016):

        fin = '{}/results/user_activity_y{}.csv'.format(proj_path,year)
        df  = pd.read_csv(fin,usecols=colnames)
        df1 = df.loc[df['n_activity']>2,:]
        if len(df1)>15000:
            df1 = df1.sample(n=15000)  # to reduce computation time
        
        X = df1.iloc[:,0:6].values
        X_scaled = preprocessing.scale(X) # standard normalization
        sse = kmeans_performance(X_scaled,kmax)
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()),  linewidth=3.0)
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        checkOutputFolder('{}/figures/kmeans'.format(proj_path))
        fn_out = '{}/figures/kmeans/ncluster_vs_sse_y{}.png'.format(proj_path,year)
        plt.savefig(fn_out)
        plt.close()

def analysis(proj_path,colnames,kval):

    for year in range(2008, 2016):
    
        fin = '{}/results/user_activity_y{}.csv'.format(proj_path,year)
        df = pd.read_csv(fin,usecols=colnames)
        df = df.loc[df['n_activity']>2,:]
        X = df.iloc[:,0:6].values
        X_scaled = preprocessing.scale(X)
        results = kmeans_analysis(X_scaled,kval)

        for gid in results:
            fig = plt.figure()
            fig.set_size_inches(12, 10)
            # axes=fig.add_subplot(111)

            
            # compute mean from native X
            mu=[np.mean(X[results[gid]['idx'],i]) for i in range(X.shape[1])]
            results[gid]['mean_X'] = mu
            plt.bar([1,2,3,4,5,6], mu, align='center', alpha=1,linewidth=3)
            plt.xticks([1,2,3,4,5,6],['a2q_src','a2q_tgt','c2q_src','c2q_tgt','c2a_src','c2a_tgt'])
         
            plt.ylabel('Count/user,N')
            plt.title('kmeans / y{}'.format(year))
            checkOutputFolder('{}/figures/kmeans'.format(proj_path))
            fn_out = '{}/figures/kmeans/kmeans_y{}_g{}_N{}.png'.format(proj_path,year,gid,results[gid]['N'])
            plt.savefig(fn_out)
            plt.close("all")


        mylist = list()
        for gid in results:
            mydat = dict(gid=gid,
                            N=results[gid]['N'],
                            m0=results[gid]['mean_X'][0],
                            m1=results[gid]['mean_X'][1],
                            m2=results[gid]['mean_X'][2],
                            m3=results[gid]['mean_X'][3],
                            m4=results[gid]['mean_X'][4],
                            m5=results[gid]['mean_X'][5]
                            )
            mylist.append(mydat)

        # Save output
        df1 = pd.DataFrame(mylist)
        checkOutputFolder('{}/results/kmeans'.format(proj_path))
        fn_csv = '{}/results/kmeans/kmeans_y{}.csv'.format(proj_path,year)
        df1.to_csv(fn_csv)



if __name__=="__main__":
    proj_path='/Users/skyeong/Desktop/stackoverflow/'
    # colnames=['a2q_src','a2q_tgt','c2a_src','c2a_tgt','c2q_src','c2q_tgt','n_activity']
    colnames=['a2q_src','a2q_tgt','c2a_src','c2a_tgt','c2q_src','c2q_tgt','deltaTS_mu','n_activity']

    # Parsing inputs
    parser = argparse.ArgumentParser(description='clusteringAnalysis_kmeans')
    parser.add_argument('-m', '--mode', help='Specify mode')
    parser.add_argument('-km', '--kmax', help='Maximum k-value for k vs. sse analysis')
    parser.add_argument('-k', '--kval', help='k-value')
    args = parser.parse_args()
    mode = args.mode

    # Check k vs sse to set the number of clusters in kmeans clustering
    if mode=='sse':
        try:
            kmax = int(args.kmax)
        except:
            kmax = 15

        analysis_k_vs_sse(proj_path,colnames,kmax)


    # kmeans analysis
    elif mode=='anal':
        try:
            kval = int(args.kval)
        except:
            kval = 4

        analysis(proj_path,colnames,kval)

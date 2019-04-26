import argparse
import os
import numpy as np
import pandas as pd
from kakaobank import checkOutputFolder
import matplotlib.pyplot as plt

def plotUserEntropy(proj_path,year):
    # Load data
    fn_csv = ('{}/results/user_entropy_y{}.csv'.format(proj_path,year))
    df = pd.read_csv(fn_csv)
    df = df.dropna()

    # Remove users who rarely contributed in stack Overflow (sum<10)
    # plot_vars=['entropy','entropy_a2q','entropy_c2q','entropy_c2a','entropy_a2q_c2q','entropy_a2q_c2a','entropy_c2q_c2a']
    plot_vars=['entropy','entropy_a2q','entropy_c2q','entropy_c2a']
    yaxis={2008:400,2009:2000,2010:4500,2011:9000,2012:15000,2013:18000,2014:18000,2015:20000}
    for varname in plot_vars:
        fig=plt.figure()
        fig.set_size_inches(10, 10)
        ax = plt.subplot(1,1,1)
        ax.set_xlim(0,8)
        ax.set_ylim(0,yaxis[year])        
        score = df[varname].values

        plt.hist(score, log=False, bins=20, edgecolor='white', density=False, facecolor='g', alpha=1)
        plt.xlabel(varname)
        plt.ylabel('Frequency')
        checkOutputFolder('{}/figures/entropy'.format(proj_path))
        fn_out = '{}/figures/entropy/entropy_y{}_{}.png'.format(proj_path,year,varname)
        plt.savefig(fn_out)


if __name__=="__main__":
    proj_path='/Users/skyeong/Desktop/stackoverflow/'
    for year in range(2008,2016):
        plotUserEntropy(proj_path,year)


        

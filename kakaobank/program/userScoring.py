import argparse
import os
import numpy as np
import pandas as pd
from kakaobank import checkOutputFolder
import matplotlib.pyplot as plt

def userScoring(proj_path,year):
    # Load data
    fn_csv = ('{}/results/user_activity_y{}.csv'.format(proj_path,year))
    df = pd.read_csv(fn_csv)

    # Influencer Score
    df['influencerScore'] = pd.Series(np.sum(np.array(df.loc[:,['a2q_src','c2a_src', 'c2q_src']]),axis=1))

    # Learner Score
    a2q_tgt = df['a2q_tgt'].values
    a2q_src = df['a2q_src'].values
    a2q_src[a2q_src==0]=1
    learnerScore = a2q_tgt/a2q_src
    df['learnerScore'] =pd.Series(learnerScore)
    print('pct(learnerScore>10) = {}'.format(100*sum(learnerScore>10)/len(learnerScore)))

    plot_vars=['influencerScore','learnerScore']
    for varname in plot_vars:
        fig=plt.figure()
        fig.set_size_inches(10, 10)

        score = df[varname].values
        log10var = np.log10(score[score>0])
        if varname=="influencerScore":
            logY = True
            ylabelName='log(Frequency)'
        else:
            logY = False
            ylabelName='Freqyency'
        plt.hist(log10var, log=logY, bins=20, edgecolor='white', density=False, facecolor='g', alpha=1)
        plt.xlabel('log10({})'.format(varname))
        plt.ylabel(ylabelName)
        checkOutputFolder('{}/figures/scoring'.format(proj_path))
        fn_out = '{}/figures/scoring/scoring_y{}_{}.png'.format(proj_path,year,varname)
        plt.savefig(fn_out)


if __name__=="__main__":
    proj_path='/Users/skyeong/Desktop/stackoverflow/'
    for year in range(2008,2016):
        userScoring(proj_path,year)


        

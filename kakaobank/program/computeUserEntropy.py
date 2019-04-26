import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
from kakaobank import getDataFromFile

def computeEntropy(src_list):
    src2trt_list = list(set(src_list['TGT'].values))
    ntrt=len(src_list['TGT'].values)
    pk=list()
    for trtid in src2trt_list:
        pk.append(sum(src_list['TGT']==trtid)/ntrt)
    entropy = -(pk*np.log(np.abs(pk))).sum()
    return entropy

def computeUserEntropy(proj_path,year):
    # Index for user collection
    start = np.int64(datetime(year, 1, 1).replace(tzinfo=timezone.utc).timestamp())
    end = np.int64(datetime(year, 12, 31).replace(tzinfo=timezone.utc).timestamp())


    # Sliding-window data selection (with duration of 2 years)
    dataTypes=['a2q','c2q','c2a']
    db=dict()
    for suffix in dataTypes:
        df = getDataFromFile(proj_path,suffix)
        db[suffix] = df.loc[(df.loc[:,'UNIXTS']>=start) & (df.loc[:,'UNIXTS']<end),:]

    # combine all data
    db1 = db['a2q']
    db1 = db1.append(db['c2a'])
    db1 = db1.append(db['c2q'])

    # find unique user_id
    users = list(set(list(db1['SRC'])))
    print('year={} / Nrow={} / N unique users={}'.format(year,len(db1),len(users)))
    
    output = list()
    cnt = 0
    for user_id in users:
        cnt += 1

        # Count the number of activity
        src_list = db1.loc[db1.loc[:,'SRC']==user_id,:]
        if len(src_list)<3:
            continue

        # Entropy for whole network
        entropy = computeEntropy(src_list)

        # Entropy for each network
        a2q_list = src_list.loc[src_list['TYPE']=='a2q',:]
        if len(a2q_list)>2:
            entropy_a2q = computeEntropy(a2q_list)
        else:
            entropy_a2q = np.nan

        c2q_list = src_list.loc[src_list['TYPE']=='c2q',:]
        if len(c2q_list)>2:
            entropy_c2q = computeEntropy(c2q_list)
        else: 
            entropy_c2q = np.nan

        c2a_list = src_list.loc[src_list['TYPE']=='c2a',:]
        if len(c2a_list)>2:
            entropy_c2a = computeEntropy(c2a_list)
        else:
            entropy_c2a = np.nan

        # Entropy for two networks 
        a2q_c2q_list = src_list.loc[(src_list['TYPE']=='a2q') | (src_list['TYPE']=='c2q'),:]
        if len(a2q_c2q_list)>2:
            entropy_a2q_c2q = computeEntropy(a2q_c2q_list)
        else:
            entropy_a2q_c2q=np.nan

        a2q_c2a_list = src_list.loc[(src_list['TYPE']=='a2q') | (src_list['TYPE']=='c2a'),:]
        if len(a2q_c2a_list):
            entropy_a2q_c2a = computeEntropy(a2q_c2a_list)
        else:
            entropy_a2q_c2a = np.nan
        
        c2q_c2a_list = src_list.loc[(src_list['TYPE']=='c2q') | (src_list['TYPE']=='c2a'),:]
        if len(c2q_c2a_list):
            entropy_c2q_c2a = computeEntropy(c2q_c2a_list)
        else:
            entropy_c2q_c2a = np.nan

        # Put data into database
        userinfo = dict(user_id = user_id, 
                        entropy = entropy,
                        entropy_a2q = entropy_a2q,
                        entropy_c2q = entropy_c2q,
                        entropy_c2a = entropy_c2a,
                        entropy_a2q_c2q = entropy_a2q_c2q,
                        entropy_a2q_c2a = entropy_a2q_c2a,
                        entropy_c2q_c2a = entropy_c2q_c2a
                        )
        # Insert user stat into DataFrame
        output.append(userinfo)
        if cnt%10000==0:
            print('   {}/{} is done'.format(cnt,len(users)))

    # concatenate dataframe
    df = pd.DataFrame(output) 

    # save output for each year    
    fn_out='{}/results/user_entropy_y{}.csv'.format(proj_path,year)
    df.to_csv(fn_out)


if __name__=="__main__":
    # Load database
    proj_path='/Users/skyeong/Desktop/stackoverflow/'

    parser = argparse.ArgumentParser(description='myprogram')
    parser.add_argument('-y', '--year', help='Select year')
    args = parser.parse_args()

    year = int(args.year)

    # Execute the method
    computeUserEntropy(proj_path,year)
    
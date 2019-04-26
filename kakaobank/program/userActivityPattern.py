import argparse
import time
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timezone
from kakaobank import getDataFromFile


def userActivityPattern(proj_path,year):

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

        # Number of total activity
        src_list = db1.loc[db1.loc[:,'SRC']==user_id,:]
        if len(src_list)<3:
            continue

        # Time Difference between successive user activity
        list_ts = np.sort(np.array(src_list['UNIXTS']))
        delta_ts = np.diff(list_ts)/60/60/24.  # in days
        deltaTS_mu = np.mean(delta_ts)
        deltaTS_sd = np.std(delta_ts)

        # Compute other user properties
        a2q_src = len(src_list.loc[src_list['TYPE']=='a2q',:])
        c2q_src = len(src_list.loc[src_list['TYPE']=='c2q',:])
        c2a_src = len(src_list.loc[src_list['TYPE']=='c2a',:])
        
        # Femove low-frequent user for each TYPE
        if a2q_src<2 | c2q_src<2 | c2a_src<2:
            continue

        # Count the number of getting comments or answers
        tgt_list = db1.loc[db1.loc[:,'TGT']==user_id,:]
        if len(tgt_list)==0:
            a2q_tgt=0
            c2q_tgt=0
            c2a_tgt=0
        else:
            a2q_tgt = len(tgt_list.loc[tgt_list['TYPE']=='a2q',:])
            c2q_tgt = len(tgt_list.loc[tgt_list['TYPE']=='c2q',:])
            c2a_tgt = len(tgt_list.loc[tgt_list['TYPE']=='c2a',:])

        # Put data into database
        userinfo = dict(user_id = user_id, 
                        n_activity = len(list_ts),
                        a2q_src    = a2q_src,
                        c2q_src    = c2q_src,
                        c2a_src    = c2a_src,
                        a2q_tgt    = a2q_tgt,
                        c2q_tgt    = c2q_tgt,
                        c2a_tgt    = c2a_tgt,
                        deltaTS_mu = deltaTS_mu,
                        deltaTS_sd = deltaTS_sd
                        )
        # Insert user stat into DataFrame
        output.append(userinfo)
        if cnt%10000==0:
            print('   {}/{} is done'.format(cnt,len(users)))

    # concatenate dataframe
    df = pd.DataFrame(output) 

    # save output for each year    
    fn_out='{}/results/user_activity_y{}.csv'.format(proj_path,year)
    df.to_csv(fn_out)


if __name__=="__main__":
    # Load database
    proj_path='/Users/skyeong/Desktop/stackoverflow/'

    parser = argparse.ArgumentParser(description='userActivityPattern')
    parser.add_argument('-y', '--year', help='Select year')
    args = parser.parse_args()

    year = int(args.year)

    # Execute analysis method
    userActivityPattern(proj_path,year)
    
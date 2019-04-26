import pandas as pd
import numpy as np
import os
from datetime import datetime
from datetime import timezone

def checkOutputFolder(dirname):
    try:
        os.stat(dirname)
        print('{} is already existed.'.format(dirname))
    except:
        os.mkdir(dirname)
        print('{} is created.',format(dirname))

def mergeUserActivity(proj_path):
    colnames=['a2q_src','a2q_tgt','c2a_src','c2a_tgt','c2q_src','c2q_tgt','n_activity']
    print('loading data saved for each year')
    for year in range(2008,2016):
        fin = '{}/results/user_activity_y{}.csv'.format(proj_path,year)
        if year==2008:
            df = pd.read_csv(fin)
        else:
            df1 = pd.read_csv(fin)
            df = df.append(df1)
    
    # To identify unique users
    user_ids = list(set(df.user_id.values))

    # Sum all data across the years
    cnt=1
    df2 = pd.DataFrame()
    print('merge 8-years data')
    for user_id in user_ids:
        df1 = df.loc[df['user_id']==user_id,:]

        userinfo = pd.DataFrame()
        userinfo.loc[1,'user_id'] = user_id
        for var in colnames:
            userinfo.loc[1,var] = sum(df1[var])

        df2 = df2.append(userinfo)
        if cnt%100000==0:
            print('  {}/{} is processing.',format(cnt,len(user_ids)))

        cnt += 1

    fn_csv = '{}/results/kmeans/kmeans_all.csv'.format(proj_path)
    df2.to_csv(fn_csv)


def getDataFromFile(proj_path=None,suffix=None):
    if proj_path is None:
        print('project path should be specified.')
        return
    if suffix is None:
        print('suffix name should be specified.')
        return
  
    # Load all data
    fin = '{}/data/sx-stackoverflow-{}.txt'.format(proj_path,suffix)
    df = pd.read_csv(fin, sep=' ', names=['SRC','TGT','UNIXTS'],header=None)  
    df['TYPE']=suffix

    return df


def getSampleDataFromFile(proj_path):
    if proj_path is None:
        print('project path should be specified.')
        return

    # Load all data
    fin = '{}/data/sample.txt'.format(proj_path)
    df = pd.read_csv(fin, sep=' ', names=['SRC','TGT','UNIXTS'],header=None)  
    df['type']='a2q'

    return df

    


def computeUserStat(db,outdir=None):
    if outdir is None:
        print('outdir path should be specified.')
    return

    output = dict()
    for year in range(2008,2017,1):
    # for year in range(2008,2017,1):
        
        # Index for user collection
        start = np.int64(datetime(year, 1, 1).replace(tzinfo=timezone.utc).timestamp())
        end = np.int64(datetime(year+1, 12, 31).replace(tzinfo=timezone.utc).timestamp())
        print('years {}-{}'.format(year,year+1))

        # Sliding-window data selection (with duration of 2 years)
        cursors = db.loc[(db.loc[:,'UNIXTS']>=start) & (db.loc[:,'UNIXTS']<end),:]
        ndoc = len(cursors)
        users = dict()
        cnt = 1
        for idx, user in cursors.T.iteritems():
            # Compute Out-degree
            uid_src = user['SRC']
            if uid_src not in list(users.keys()):
                users[uid_src] = {'a2q_src':0,'a2q_tgt':0,'c2q_src':0,'c2q_tgt':0,'c2a_src':0,'c2a_tgt':0}
            key_src='{}_src'.format(user['type'])
            users[uid_src][key_src] += 1
            
            # Compute In-degree
            uid_tgt = user['TGT']
            if uid_tgt not in list(users.keys()):
                users[uid_tgt] = {'a2q_src':0,'a2q_tgt':0,'c2q_src':0,'c2q_tgt':0,'c2a_src':0,'c2a_tgt':0}
            key_tgt='{}_tgt'.format(user['type'])
            users[uid_tgt][key_tgt] += 1
            
            # Increment the index for user collection 
            # (each sliding window has unique user index)
            if ndoc % 10000 == 0:
                print('   {}/{} is loaded.'.format(cnt,ndoc))
            cnt += 1
        df = pd.DataFrame(users).T

        if outdir is not None:
            fn_out='{}/data/user_snapshot_{}.csv'.format(outdir,year)
            df.to_csv(fn_out)
        else:
            output[year]=df

    if outdir is None:
        return output
 


if __name__=='__main__':

    proj_path='/Users/skyeong/Desktop/stackoverflow/'

    # Load data from file
    db = getDataFromFile(proj_path)

    # db = getSampleDataFromFile(proj_path)
    computeUserStat(db,proj_path)
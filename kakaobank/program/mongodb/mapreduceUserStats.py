import csv
import time
import networkx as nx
import pandas as pd
from datetime import datetime
from datetime import timedelta
from pymongo import MongoClient

from bson.code import Code
mapSRC = Code("function () {emit(this.SRC, 1);}")
mapTGT = Code("function () {emit(this.TGT, 1);}")
reduce = Code("function (key, values) {"
    "  var total = 0;"
    "  for (var i in values) {"
    "    total += 1;"
    "  }"
    "  return total;"
    "}")
# Load database
proj_path='/Users/skyeong/Desktop/stackoverflow/data/'
db = MongoClient().kakaobank
# start = datetime(2008, 8, 1)
# end = datetime(2009, 7, 31)

# Converting Year to Index
y2i = {2008:0,2009:1,2010:2,2011:3,2012:4,2013:5,2014:6,2015:7}
types = ['a2q','c2a','c2q']
for year in range(2008,2016,1):

    start = datetime(year, 1, 1).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    end   = datetime(year, 12, 31).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    print('Data from {} to {} are processing.'.format(start,end))

    # Map-reduce to compute in-degree and out-degree for each user
    a2q_src = db.dynet.map_reduce(mapSRC, reduce, "myresults1", query={'UTCTS': {'$gte': start, '$lt': end},'type':'a2q'})
    a2q_tgt = db.dynet.map_reduce(mapTGT, reduce, "myresults2", query={'UTCTS': {'$gte': start, '$lt': end},'type':'a2q'})
    c2q_src = db.dynet.map_reduce(mapSRC, reduce, "myresults3", query={'UTCTS': {'$gte': start, '$lt': end},'type':'c2q'})
    c2q_tgt = db.dynet.map_reduce(mapTGT, reduce, "myresults4", query={'UTCTS': {'$gte': start, '$lt': end},'type':'c2q'})
    c2a_src = db.dynet.map_reduce(mapSRC, reduce, "myresults5", query={'UTCTS': {'$gte': start, '$lt': end},'type':'c2a'})
    c2a_tgt = db.dynet.map_reduce(mapTGT, reduce, "myresults6", query={'UTCTS': {'$gte': start, '$lt': end},'type':'c2a'})
    
    # Insert it into dataframe
    df1 = pd.DataFrame(list(a2q_src.find())).set_index('_id')
    df2 = pd.DataFrame(list(a2q_tgt.find())).set_index('_id')
    df3 = pd.DataFrame(list(c2q_src.find())).set_index('_id')
    df4 = pd.DataFrame(list(c2q_tgt.find())).set_index('_id')
    df5 = pd.DataFrame(list(c2a_src.find())).set_index('_id')
    df6 = pd.DataFrame(list(c2a_tgt.find())).set_index('_id')
    
    # concatenate dataframe
    df = pd.concat([df1, df2, df3, df4, df5, df6], axis=1)
    df.index = pd.Int64Index(df.index)
    df.columns=['a2q_src','a2q_tgt','c2q_src','c2q_tgt','c2a_src','c2a_tgt']
    df = df.fillna(0)
    
    fn_out='{}/user_snapshot_{}.csv'.format(proj_path,year)
    df.to_csv(fn_out)

    # Insert it into database
    # idx = y2i[year]
    # db.user[idx].insert_many(df.to_dict())
   
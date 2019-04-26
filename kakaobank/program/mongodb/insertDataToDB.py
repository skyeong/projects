import csv
import time
from datetime import datetime
from pymongo import MongoClient

def insertDataToDB(proj_path):
    # Create empty database
    db = MongoClient().kakaobank

    # Raw data path
    fnames = ['a2q','c2a','c2q']

    # Insert dynamic graph into the database
    start_time = time.time()
    for suffix in fnames:
        fin = '{}/data/sx-stackoverflow-{}.txt'.format(proj_path,suffix)
        with open(fin) as csvDataFile:
            csvReader = csv.reader(csvDataFile)
            counter = 0
            for row in csvReader:
                if len(row) != 0:
                    arow = [int(dat) for dat in row[0].split(' ')]
                    
                    # convert Unix timestamp to UTC time
                    UTCTS = datetime.utcfromtimestamp(arow[2]).strftime('%Y-%m-%dT%H:%M:%S.000Z')
                    
                    # Insert data into db
                    result = db.dynet.insert_one({"type":suffix,"SRC": arow[0], "TGT":arow[1], "UTCTS":UTCTS})
                    counter += 1

            print("{} rows from {} were inserted in db.".format(counter,fin))

    elapsed_time = time.time() - start_time
    print("Total number of inserted data is {}".format(db.dynet.count()))
    print('Total elapsed time to load and insert them into database is {}s.'.format(elapsed_time))


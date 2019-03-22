import platform
import unicodecsv
import pandas as pd

def read_csv(csv_filename, encoding='utf-8'):
    with open(csv_filename, 'rb') as csv_file:
        if platform.system() == 'windows':
            encoding='utf-8-sig'
        reader = unicodecsv.DictReader(open(csv_filename, 'rb'), encoding=encoding)
        data = [row for row in reader]
    return data



def write_csv(datalist, f, encoding='utf-8'):
    filenames = datalist[0].keys()
    headerinfo = dict([(v,v) for v in filenames])
    writer = unicodecsv.DictWriter(open(f, 'wb'), filenames=filenames, encoding=encoding)
    writer.writerow(headerinfo)
    with open(f,'wb') as output:
        for row in datalist:
            writer.writerow(row)
import json
import math
from urllib.request import urlopen
from urllib.parse import urlencode
from datetime import datetime
from datetime import timedelta

class BoxOffice(object):
    base_url='http://www.kobis.or.kr/kobisopenapi/webservice/rest/boxoffice/searchDailyBoxOfficeList.json'

    def __init__(self, api_key):
        self.api_key = api_key
    
    def get_movies(self):
        target_dt = datetime.now() - timedelta(days=1)
        target_dt_str = target_dt.strftime('%Y%m%d')
        query_url = '{}?key={}&targetDt={}'.format(self.base_url, self.api_key,target_dt_str)
        with urlopen(query_url) as fin:
            return json.loads(fin.read().decode('utf-8'))

    def simplify(self, result):
        return [
            {
                'rank': entry.get('rank'),
                'name': entry.get('movieNm'),
                'code': entry.get('movieCd')
            }
            for entry in result.get('boxOfficeResult').get('dailyBoxOfficeList')
        ]

api_key='f6270d78f25bc180d62db5c274cebe7c'
box = BoxOffice(api_key)
movies = box.get_movies()
print(box.simplify(movies))
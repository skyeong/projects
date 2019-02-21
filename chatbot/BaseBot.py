import json
import requests
import urllib
import time

class BaseBot:
    def __init__(self, token_name):
        TOKEN = open(token_name,'r').read()
        bot_url = "https://api.telegram.org/bot{}/".format(TOKEN)
        self.bot_url = bot_url

    def get_updates(self,offset=None):
        url = self.bot_url + "getUpdates"
        if offset:
            url += "?offset={}".format(offset)
        # print(url)
        response = requests.get(url)
        content = response.content.decode("utf8")        
        js = json.loads(content)
        return js

    def get_last_update_id(self,updates):
        update_ids = []
        for update in updates["result"]:
            update_ids.append(int(update["update_id"]))
        return max(update_ids)

    def get_last_chat_id_and_text(self,updates):
        num_updates = len(updates["result"])
        last_update = num_updates - 1
        text = updates["result"][last_update]["message"]["text"]
        chat_id = updates["result"][last_update]["message"]["chat"]["id"]
        return (text, chat_id)

    def send_welcome_message(self,chat_id):
        message = "안녕하세요^^ 여러분의 스트레스를 관리해주는 스봇입니다. 만나서 반가워요~"
        self.send_message(message,chat_id)
        
    def send_message(self, text, chat_id):
        text = urllib.parse.quote_plus(text)
        url = self.bot_url + "sendMessage?text={}&chat_id={}".format(text, chat_id)
        print(url)
        requests.get(url)

    def listening(self, last_update_id=None):
        while True:
            updates = self.get_updates(last_update_id)
            if len(updates["result"])>0:
                last_update_id = self.get_last_update_id(updates)+1
                print('update id:{}'.format(last_update_id))
                
                # Loop over each new message
                for update in updates["result"]:
                    try:
                        content = update["message"]["text"]
                        chat_id = update["message"]["chat"]["id"]
                    except Exception as e:
                            print(e)
                            continue
                    
                    # 상황에 따른 대처
                    if content == '/start':
                        self.send_welcome_message(chat_id)
                    elif content == '/stop':
                        print('user_id {} is out of sbot'.format(chat_id))
                    elif content.startswith('/show'):
                        print('show menu')
                    else: # just echo message
                        self.send_message(content, chat_id)

            time.sleep(0.5)

def main():
    token_name='sbot.token'
    bot = BaseBot(token_name) 

    last_update_id = None
    bot.listening(last_update_id)
  

if __name__=="__main__":
    main()
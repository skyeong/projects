import time
import requests
from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey

API_KEY = "WnNwYZKv5IPS7oMoL9aOsLK1lXWfMEnw1ljNDpyWuoGR"
ACCOUNT  = "5f72cdcf-65c5-462e-b8ae-1e5a64c0091c-bluemix"
PASSWORD= "b9671d753a854e5efbbfb59b10194caf878c69b3"

def get_access_token(api_key):
    """Retrive an access token from the IAM token service."""
    token_response = requests.post(
        "https://iam.bluemix.net/oidc/token",
        data={
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "response_type": "cloud_iam",
            "apikey": api_key        
        },
        headers={
            "Accept": "application/json"
        }
    )
    if token_response.status_code == 200:
        print("Got access token from IAM")
        return token_response.json()["access_token"]
    else:
        print("{} {}\n".format(token_response.status_code,token_response.json()))
    return None


def main(api_key, account):
    access_token = None
    while True:
        if not access_token:
            access_token = get_access_token(api_key)

        if access_token:
            response = requests.get(
                "https://{0}.cloudant.com/_all_dbs".format(account),
                headers={
                    "Accept": "application/json",
                    "Authorization": "Bearer {0}".format(access_token)
                }
            )
            print("Got Cloudant response, status code", response.status_code)
            if response.status_code == 401:
                print ("Token has expired.")
                access_token = None

        time.sleep(1)

if __name__=="__main__":
    #main(API_KEY, ACCOUNT)
    serviceUsername = ACCOUNT
    servicePassword = PASSWORD
    serviceURL = "https://{}.cloudant.com".format(ACCOUNT)

    client = Cloudant(serviceUsername, servicePassword, url=serviceURL)
    client.connect()


    # Creating database
    databaseName = "databasedemo"
    myDatabaseDemo = client.create_database(databaseName)

    if myDatabaseDemo.exists():
        print("'{0}' successfully created.\n".format(databaseName))

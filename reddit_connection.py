import requests
import os
from dotenv import load_dotenv


load_dotenv(dotenv_path='keyword.env')
client_id = os.getenv('CLIENT_ID')
client_secret = os.getenv('CLIENT_SECRET')
password = os.getenv('PASSWORD')

auth = requests.auth.HTTPBasicAuth(client_id,client_secret)
data = {
  'grant_type': 'password',
  'username': 'u/One_Competition4014',
  'password': password
}

headers = {'User-Agent':'TccTeste/0.0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

res.json()
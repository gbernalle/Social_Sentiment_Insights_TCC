import requests
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='keyword.env')
client_id = os.getenv('CLIENT_ID')
secret_key = os.getenv('SECRET_KEY')
password = os.getenv('PASSWORD')
user_name = os.getenv('USER_REDDIT')

auth = requests.auth.HTTPBasicAuth(client_id, secret_key)
data = {
  'grant_type': 'password',
  'username': user_name,
  'password': password
}

headers = {'User-Agent':'TccTeste/0.0.1'}

res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers, verify=False)

TOKEN = res.json()['access_token']
headers['Authorization'] = f'bearer {TOKEN}'

requests.get('https://oauth.reddit.com/api/v1/me',headers=headers, verify=False).json()
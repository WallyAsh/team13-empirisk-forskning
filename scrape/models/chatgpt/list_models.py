import os
import openai
from dotenv import load_dotenv

load_dotenv()  # This loads variables from your .env file
openai.api_key = os.getenv("OPENAI_API_KEY")

models = openai.Model.list()
for model in models['data']:
    print(model['id'])

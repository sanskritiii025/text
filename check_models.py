import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

# 1. Provide your actual key
MY_API_KEY = os.getenv("GOOGLE_API_KEY")   

# 2. Connect to Google's API
client = genai.Client(api_key=MY_API_KEY)

print("Fetching available embedding models from Google...\n")

# 3. Ask the server for a list of all models and filter for embeddings
for model in client.models.list():
    if "embedding" in model.name.lower() or "embed" in model.name.lower():
        print(f"Found Model Name: {model.name}")
import os

google_api_key = os.environ.get('GOOGLE_API_KEY', 'AIzaSyAtNQ0N9i16Ad3iBiHPLRJq2OJj5T79P6g')
repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
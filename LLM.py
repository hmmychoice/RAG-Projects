import os
import config
from langchain_community.llms import HuggingFaceEndpoint


#Embedding Model
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name = config.model_name)

#LLM
llm = HuggingFaceEndpoint(
    repo_id=config.repo_id, max_length=128, 
    temperature=0.8, 
    token=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
)

#prompt template
template = """Provide the answers based on the context given. 
Do not make up the things and just respond "Do not have enough knowledge"
Question: {question} 
Context: {summaries}
Answer:
"""
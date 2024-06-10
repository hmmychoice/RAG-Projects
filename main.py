import os
import LLM
import streamlit
import time
import streamlit as st
import pickle
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate

st.title("News Research Tool (RAG)")
st.sidebar.title("Provide News Article URL")

urls = []
url = st.sidebar.text_input(f"URL")
urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_db.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1200,
        chunk_overlap = 200
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)


    # create embeddings and save it to FAISS index
    embeddings = LLM.embeddings
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_chain_type(llm=LLM.llm,
                                                                chain_type="stuff",
                                                                retriever=vectorstore.as_retriever(),
                                                                chain_type_kwargs={
                                                                            "prompt": PromptTemplate(
                                                                                template=LLM.template,
                                                                                input_variables=["summaries", "question"],
                                                                            ),
                                                                        })
            
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)





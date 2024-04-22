from langchain import HuggingFaceHub
from langchain.llms import HuggingFaceHub

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os
#os.environ["OPENAI_API_KEY"] = "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"

# Initialize session state to store chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    # About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Hugging Face](https://huggingface.co/) LLM model
 
    ''')
    #add_vertical_space(5)
    st.write('')

load_dotenv()
def main():
  #st.write("Hello !!")
  st.header("Chat with PDF 💬")

  # upload a PDF file
  pdf = st.file_uploader("Upload your PDF", type='pdf')

  # st.write(pdf)
  if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
      text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )

    chunks = text_splitter.split_text(text=text)
    #st.write(chunks)

    # # embeddings
    store_name = pdf.name[:-4]
    #st.write(f'{store_name}')

    if os.path.exists(f"{store_name}.pkl"):
      with open(f"{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
      #st.write("Embeddings Loaded from the Disk")


    else:
      embeddings = HuggingFaceEmbeddings()
      VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
      with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(VectorStore, f)
      #st.write("Embeddings computation completed")

    #User Query
    query = st.text_input("Ask questions about your PDF file:")
    st.write(query)

    if query:
      docs = VectorStore.similarity_search(query=query, k=3)

      #google/flan-t5-xxl
      #google/flan-t5-large

      llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.8, "max_length": 512},huggingfacehub_api_token="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")


      #llm = "sentence-transformers/all-mpnet-base-v2"
      chain = load_qa_chain(llm=llm, chain_type="stuff")

      response = chain.run(input_documents=docs, question=query)
      #print(cb)
      #st.write(response)
      # Save user query and response to chat history
      st.session_state.messages.append({"role": "user", "content": query})
      st.session_state.messages.append({"role": "assistant", "content": response})

      # Display chat history
      for message in st.session_state.messages:
          if message["role"] == "user":
            st.text("User: " + message["content"])
          elif message["role"] == "assistant":
            st.text("Assistant: " + message["content"])



if __name__ == '__main__':
  main()
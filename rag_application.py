import streamlit as st
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
load_dotenv()

from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

key=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=key)
model=genai.GenerativeModel('gemini-2.0-flash')

@st.cache_resource(show_spinner="Loading embedding model...")

def load_embedding_model():
    return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

embed_model=load_embedding_model()

st.header('RAG Assistant')
st.subheader('RAG Assistance for Document Summary')

file=st.file_uploader('upload your pdf document',type=['pdf'])

if file:
    raw_text=''
    reader=PdfReader(file)
    for page in reader.pages:
        text=page.extract_text()
        if text:
            raw_text +=text
    if raw_text.strip():
        doc=Document(page_content=raw_text)
        char_split=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks=char_split.split_documents([doc]) # documents gets converted into small chunck
        
        txt=[i.page_content for i in chunks]    
           
               
        vectordb=FAISS.from_texts(txt,embed_model)
        retrive=vectordb.as_retriever()
        st.success('Document processed sucessfully')
        
        query=st.text_input('Enter your question')
        
        if query:
            with st.chat_message('ai'):
                
                with st.spinner('Analysing the document....'):
                    query_response=retrive.get_relevant_documents(query)
            
                    context="\n\n".join([i.page_content for i in query_response])
            
                    prompt=f'''You are the ai assistant use the content extracted {context}
                    and answer the query {query}asked and display the output if you do not know
                    about the context tell me Not Known to you'''
            
                    response=model.generate_content(prompt)
                    st.markdown('Answer')
                    st.write(response.text)
    else:
        st.warning('No text could be read submit the valid document')
            
            
        
            
            
        
        
        
        
    
    
    
        
        





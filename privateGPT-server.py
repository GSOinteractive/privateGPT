from flask import Flask,jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import chromadb
import os
import argparse
import time

from constants import CHROMA_SETTINGS

app = Flask(__name__)
CORS(app)

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS
llm = None
    
@app.route('/get_answer', methods=['POST'])
def get_answer():
    global llm
    query = request.json

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever()

    if llm==None:
        return "Model not downloaded", 400    

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    if query!=None and query!="":
        res = qa(query)
        answer, docs = res['result'], res['source_documents']
        
        source_data =[]
        for document in docs:
             source_data.append({"name":document.metadata["source"]})

        return jsonify(query=query,answer=answer,source=source_data)

    return "Empty Query",400

def load_model():
    global model_type
    global model_path
    global model_n_ctx
    global model_n_batch
    global llm

    callbacks = [StreamingStdOutCallbackHandler()]

    match model_type:
        case "LlamaCpp":
            print("load LLM model from path " + model_path)
            llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case "GPT4All":
            print("load GPT4All model from path " + model_path)
            llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")

    
if __name__ == "__main__":
  load_model()
  print("LLM0", llm)
  app.run(host="0.0.0.0", debug = False)

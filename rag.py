import os
from flask import Flask, request, jsonify
from langchain import hub
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from slack_bolt.adapter.flask import SlackRequestHandler
import os
from slack_bolt.app import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler


# This is from our embedding model
VECTOR_DIMENSION = 768
INDEX_NAME = "thtpledgeinfo"

load_dotenv()

# GROQ_API_KEY needs to be an environment variable (create a .env file for this)
key = os.getenv(key="GROQ_API_KEY")
from langchain_groq import ChatGroq
llm = ChatGroq(model="llama3-8b-8192",api_key=key)

loader = DirectoryLoader('docs/', glob="**/*.txt")
docs = loader.load()

embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
# May want to play around with chunking sizes to make it run better
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Pinecone version -------
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-east-1-aws')
pc = Pinecone(os.getenv(key="PINECONE_API_KEY"))
spec = ServerlessSpec(cloud='aws', region='us-east-1')  

print("Creating vectorstore...")
# vectorstore = PineconeVectorStore.from_documents(
#     documents=splits,
#     index_name=INDEX_NAME,
#     embedding=embed
# )
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embed
)
print("Vectorstore Created.")
#-------------------------

# Retrieve and generate using the relevant snippets of THT info.
retriever = vectorstore.as_retriever()
print("Retriever created.")

# # We can probably change this at some point
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = ( 
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)



def ask_tht_ai(input_prompt)->str:
    print("Invoking chain...")
    out = ""
    for chunk in rag_chain.stream(input_prompt):
        print("Streaming...")
        out+=chunk
    return out


def add_docs(directory):
    try:
        loader = DirectoryLoader('docs/', glob="**/*.txt")
    except:
        print("Invalid directory or documents. Please input filepath to directory with .txt files.")
        return
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    try:
        vectorstore.add_documents(splits)
    except:
        print("Issue adding documents to vectorstore. Operation was cancelled.")
    
# def lambda_handler(event, context):
#     return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }

#################################################################3

# Initializes your app with your bot token and signing secret
app = App(
    token=os.getenv(key="SLACK_BOT_TOKEN"),
    signing_secret=os.getenv(key="SIGNING_SECRET")
)

flask_app = Flask("ChatTHT")

# Slash command handler
@app.command("/askiggy")
def handle_hello_command(ack, body, respond):
    # Acknowledge the command request
    ack()
    
    message = body["text"]
    respond(ask_tht_ai(message))

# Start your app
handler = SlackRequestHandler(app) #SocketModeHandler(app=app, app_token=os.getenv(key="SLACK_APP_TOKEN"))

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    # Slack verifies the URL before sending events
    return handler.handle(request)

@flask_app.route("/slack/verify", methods=["GET"])
def slack_verify():
    return "Slack verification successful!"

                          
flask_app.run(port=3000)
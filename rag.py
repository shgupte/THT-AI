import os
import time
from langchain import hub
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

# ChromaDB version
# vectorstore = Chroma.from_documents(documents=splits, embedding=embed)

# Pinecone version -------
# pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment='us-east-1-aws')
pc = Pinecone(os.getenv(key="PINECONE_API_KEY"))
spec = ServerlessSpec(cloud='aws', region='us-east-1')  

# This is all code for creating an index - 
# when the index does not initially exist, the app doesn't really work

# if pc.has_index(INDEX_NAME):  
# print("Deleting...")
# try:
#     pc.delete_index(INDEX_NAME)  
# except:
#     print("Could not delete index.")
# create a new index  

# print("Creating index...")
# pc.create_index(  
#     INDEX_NAME,  
#     dimension=VECTOR_DIMENSION,  # dimensionality of text-embedding-ada-002  
#     metric='cosine',  
#     spec=spec  
# )  
# wait for index to be initialized  
# while not pc.describe_index(INDEX_NAME).status['ready']:  
#     print("Working...")
#     time.sleep(1)

print("Creating vectorstore...")
vectorstore = PineconeVectorStore.from_documents(
    documents=splits,
    index_name=INDEX_NAME,
    embedding=embed
)
print("Vectorstore Created.")
#-------------------------

# Retrieve and generate using the relevant snippets of THT info.
retriever = vectorstore.as_retriever()
print("Retriever created.")
# query = "Who can help us with the CS pledge project?"
# host = pc.describe_index(INDEX_NAME).host
# index = pc.Index(index_name=INDEX_NAME, host=host)

# results = vectorstore.similarity_search(
#     "Who should I ask for help for the CS pledge project",
#     k=2,
# )
# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")

# query_vec = embed.embed_query(query)

# results = index.query(vector=query_vec, top_k=1)
# if not results.matches:
#     print("No matches found.")
# else:
#     for match in results.matches:
#         print(f"Match score: {match.score}, Match vector ID: {match.id}")

# retrieved_docs = retriever.invoke(query)  # Try retrieving top 5 results
# if not retrieved_docs:
#     print("No documents retrieved.")
# else:
#     print("Retrieving docs...\n")
#     print(f"{len(retrieved_docs)} documents retrieved")
#     for doc in retrieved_docs:
#         print(doc.page_content[:500])  # Print a snippet of each retrieved doc

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
    

# print(ask_tht_ai(input("Enter your prompt: ")))
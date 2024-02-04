import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# setup
st.set_page_config(page_title="Chat with any website!!!")
st.title("Chat with Websites")

def url_to_vectorstore(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # split documents into chunks
    # each chunk also has its metadata, which is useful for identification
    # if we were talking to multiple websites, it would be useful
    # for our purposes since we are talking to one website it is not useful
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    # create a vectorstore from chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store

# returns data chunks relevant
def get_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# get response from AI model
def get_response(query):
    # create conversation chain
    # vector_store = url_to_vectorstore(url)
    retriever_chain = get_retriever_chain(st.session_state.vector_store) # with old code vector_store would go in parentheses
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input": query
        })
    return response["answer"]

# sidebar
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Website URL")

if url is None or url == "":
    st.info("Please enter a website URL")
else:
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am an AI bot. How can I help you?"),
            ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = url_to_vectorstore(url)
    
    query = st.chat_input("Type a message!") # user input
    if query is not None and query != "":
        response = get_response(query)
        # st.write(response)         <- this would be good for looking at the response object
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation between human and chatbot
    # loop through chat history in session state
    # if the item is an instance of AIMessage, write the AI message
    # if the item is an instance of HumanMessage, write the Human message
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)








import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# setup
st.set_page_config(page_title="Chat with any website!!!")
st.title("Chat with Websites")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am an AI bot. How can I help you?"),
        ]

# get response from AI model
def get_response(query):
    return "AODNS"

# sidebar
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Website URL")

if url is None or url == "":
    st.info("Please enter a website URL")
else:
    query = st.chat_input("Type a message!") # user input
    if query is not None and query != "":
        response = get_response(query)
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation between human and chatbot
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)








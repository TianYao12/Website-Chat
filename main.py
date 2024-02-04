import streamlit as st

st.set_page_config(page_title="Chat with any website!!!")

st.title("Chat with Websites")

with st.sidebar:
    st.header("Settings")
    url = st.text_input("Website URL")
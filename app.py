import streamlit as st
from src.NMT import NMT

# Initialize your custom model
custom_model = NMT()

# Streamlit UI
with st.sidebar:
    st.title('🤖💬 Financial Advisor Assistant')
    language = None
    #language = st.selectbox("Select language:", ["Hindi", "Spanish", "French", 'German'])

if "messages" not in st.session_state:
    st.session_state.messages = []
    
    

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner('Thinking...'):
        response = custom_model.text_generation(prompt, language=language)
        AI_text = response['text']
    st.session_state.messages.append({"role": "assistant", "content": AI_text})
    with st.chat_message("assistant"):
        st.markdown(AI_text)

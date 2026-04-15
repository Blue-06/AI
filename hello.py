######### ---------- Please do not modify this code here. ---------- ##########
#### Download a copy of this code to your local machine to modify & test.  ####
####-----------------------------------------------------------------------####

#pip install langchain-openai 
from langchain_openai import ChatOpenAI  
import os  
import httpx  
import streamlit as st

client = httpx.Client(verify=False) 

st.title("Welcome to GenAi - Team 07")


llm = ChatOpenAI( 
    openai_api_base="https://genailab.tcs.in",
    model = "azure_ai/genailab-maas-DeepSeek-V3-0324", 
    api_key="na kr ladle",
    # Will be provided during event.  And this key is for Hackathon purposes only
    # and should not be used for any unauthorized purposes 
    http_client = client 
) 
user_input = st.text_input("Enter your prompt:")

if user_input:  # Only run if user typed something
    if st.button("Send"):
        with st.spinner("Generating response..."):
            response = llm.invoke(user_input)
        st.success("Response received!")
        st.write("### Model Output:")
        st.write(response.content)
else:
    st.info("👆 Type a prompt above and click **Send** to see the model’s reply.")


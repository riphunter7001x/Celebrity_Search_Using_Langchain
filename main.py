# integrate our code with openai
from connstants import openai_key 
import os
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"] = openai_key

# stramlit framework

st.title("Celebrity Search Result")


# taking input 
input_text = st.text_input("enter name of celebrity or any famous personality!")
# prompt template
input_prompt_1 = PromptTemplate(
    input_variables=["name"],
    template="tell me about celebrity {name} " 
)
llm = OpenAI(temperature=0.5)
chain = LLMChain(llm=llm,prompt=input_prompt_1,verbose= True,output_key ="person" )

# prompt template
input_prompt_2 = PromptTemplate(
    input_variables=["person"],
    template="when was {person} born. in DD/MM/YYYY format" 
)
chain2 = LLMChain(llm=llm,prompt=input_prompt_2,verbose= True,output_key ="Date_of_birth")

# prompt template
input_prompt_3 = PromptTemplate(
    input_variables=["Date_of_birth"],
    template="Mention 5 major events happened around on {Date_of_birth}. in the world " 
)
chain3 = LLMChain(llm=llm,prompt=input_prompt_3,verbose= True,output_key ="decription")


perent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables = ["name"],output_variables =["person", "Date_of_birth","decription"],verbose= True)

# taking input
if input_text: 
    st.write(perent_chain({"name":input_text}))

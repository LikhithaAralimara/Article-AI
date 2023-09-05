#bring in deps
import os 
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain , SimpleSequentialChain , SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey 

#app framework
st.subheader('LIKHITHA ARALIMARA presents')
st.title('🦜️🔗 Article AI ')
st.subheader('Transform Topics into Articles with a Single Tap')
prompt = st.text_input('Plug in your prompt here')

#Prompt templates
title_template = PromptTemplate(
    input_variables =['topic'],
    template='write an article title about {topic}'
)

script_template = PromptTemplate(
    input_variables =['title'],
    template='write a very short article script based on this title {title}'
)

 #memory
memory = ConversationBufferMemory(input_key='topic' , memory_key='chat history')

#llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm , prompt = title_template , verbose=True,output_key='title',memory=memory)
script_chain = LLMChain(llm=llm , prompt = script_template , verbose=True,output_key='script',memory=memory)
#sequential_chain=SimpleSequentialChain(chains=[title_chain,script_chain],verbose=True)
sequential_chain=SequentialChain(chains=[title_chain,script_chain],verbose=True,input_variables=['topic'],output_variables=['title' , 'script'])


#show stuff to the screen if the prompt is run
if prompt:
    #for gettign only title
    #response = title_chain.run(topic=prompt)
    #st.write(response)

    #for getting only script (although title is generated its not displayed/printed)
    #response = sequential_chain.run(prompt)
    #st.write(response)

    #to get both title and script
    response= sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
       st.info(memory.buffer)

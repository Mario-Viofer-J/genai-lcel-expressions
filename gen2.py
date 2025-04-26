#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[2]:


from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


# In[9]:


prompt = ChatPromptTemplate.from_template(
    "tell me a short story about {topic} with {number} of words in the story"
)
model = ChatOpenAI()
output_parser = StrOutputParser()


# In[10]:


chain = prompt | model | output_parser


# In[11]:


chain.invoke({"topic": "lions","number":"90"})


# In[12]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch


# In[14]:


vectorstore = DocArrayInMemorySearch.from_texts(
    ["issac newton invented gravity", "thomas edison invented light and electricity"],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


# In[ ]:





# In[15]:


retriever.get_relevant_documents("what did thomas edison invented?")


# In[16]:


retriever.get_relevant_documents("what did issac newton invented?")


# In[17]:


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# In[18]:


from langchain.schema.runnable import RunnableMap


# In[19]:


chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


# In[20]:


chain.invoke({"question": "who is the invented light?"})


# In[21]:


inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})


# In[22]:


inputs.invoke({"question": "who is issac newton?"})


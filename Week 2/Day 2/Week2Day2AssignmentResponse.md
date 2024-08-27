# Question 1
What other models could we use, and how would the above code change?

Other models available 
   gpt-4o-mini
   gpt-4o-mini-2024-07-18
   gpt-4-turbo
   gpt-4-turbo-2024-04-09 ....
   dall-e-2 & dall-e-3

```python
from langchain_openai import ChatOpenAI
openai_chat_model = ChatOpenAI(model="<model_names>")
```

# Acitivty 1

 1. Semantic Chunking


# Question 2
What is the embedding dimension, given that we are using `text-embedding-3-small`?
1536

# Question 3
What does LCEL do that makes it more reliable at scale?

- Chaining the output of one component with other components.
- Configure retries and fallbacks for any part of your LCEL chain. This is a great way to make your chains more reliable at scale.


# Day 2
# Activity 1: Include a screenshot of your trace and explain what it means

```python

base_rag_prompt_template = """\
You are a helpful assistant that can answer questions related to the provided context. Repond I don't have that information if outside context.

Context:
{context}

Question:
{question}
"""

base_rag_prompt = ChatPromptTemplate.from_template(base_rag_prompt_template)

base_llm = ChatOpenAI(model="gpt-4o-mini", tags=["base_llm"])

retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<<SOME USER QUESTION>>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": base_rag_prompt | base_llm, "context": itemgetter("context")}
)

retrieval_augmented_qa_chain.invoke({"question" : "What is LangSmith?"}, {"tags" : ["Demo Run"]})['response']
```


Langsmith Interface Walkthrough: https://www.loom.com/share/c87f632bd715428aa04dcba82f9b88cc?sid=6ceec49b-fb32-4689-b837-82d83b7b438a


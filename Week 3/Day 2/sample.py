from langchain_core.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_openai.chat_models import ChatOpenAI
from langchain.document_loaders import PyMuPDFLoader
from typing import Any
from langchain_community.vectorstores import Qdrant
import asyncio

class FetchArxivInput(BaseModel):
    "The input for the `fetch_arxiv` tool."

    query: str = Field(
        description="Adapted version of the user query to work better with ArXiv. This must be optimized to get the most relevant results."
    )


@tool("fetch_arxiv", args_schema=FetchArxivInput)
async def fetch_arxiv(query: str):

    # Get relevant ids
    docs = ArxivAPIWrapper().get_summaries_as_docs(query=query)
    arxiv_ids = [doc.metadata.get("Entry ID").split("/")[-1] for doc in docs]
    print(arxiv_ids)

    # Fetch docs
    papers = await asyncio.gather(*[PyMuPDFLoader(file_path=f"https://arxiv.org/pdf/{id}").aload() for id in arxiv_ids])

    return papers
---



# Dynamic arxiv paper retrival and loading to qdrant

arxiv_query = ArxivQueryRun()
arxiv_loader = ArxivLoader()
tool_belt = [arxiv_query]

# Initialize the OpenAI model
llm_find_title = ChatOpenAI(model="gpt-4", temperature=0)
llm_find_title = llm_find_title.bind_tools(tool_belt)

# Define the prompt for extracting the paper title
extract_title_prompt_template = """
QUERY:
{question}

Your task is to extract the title of the paper from the given user question and use `arxiv_query` tool to find the paper title.
Return only the title of the paper.
"""
extract_title_prompt = ChatPromptTemplate.from_template(extract_title_prompt_template)

qdrant_client = QdrantClient(location=":memory:")

# Function to add documents to Qdrant
def add_documents_to_qdrant(paper_title):
    arxiv_loader = ArxivLoader(query=paper_title)
    docs = arxiv_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    qdrant_arxiv_vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding_model=embedding_model,
        location=":memory:",
        collection_name=paper_title,
        client=qdrant_client
    )
    return qdrant_arxiv_vectorstore

def call_tool(ai_message: AIMessage):
    for tool in ai_message.tool_calls:
        if tool["name"] == "arxiv":
            return arxiv_query.invoke(tool)


# Create the LLM chain for extracting the paper title
extract_title_chain = extract_title_prompt | llm_find_title | call_tool 
extract_title_chain.invoke({"question": question})

# rag_chain_arxiv = (
#     {"question": itemgetter("question"), "paper_title": itemgetter("question") | extract_title_chain}
#     | 
#     {"context": itemgetter("question") | extract_title_chain , "question": itemgetter("question")}
#     | rag_prompt | openai_chat_model | StrOutputParser()
# )

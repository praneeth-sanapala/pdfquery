import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub

# Set environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = END_POINT
os.environ["AZURE_OPENAI_API_KEY"] = KEY

# Initialize embeddings and language model
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=EMBEDDING_DEPLOYMENT_NAME,
    openai_api_version=VERSION,
)

llm_azure = AzureChatOpenAI(
    azure_deployment=GPT_4O_DEPLOYMENT_NAME,
    openai_api_version=VERSION,
    temperature=0
)

# Load documents and prepare text splitter
documents = PyPDFDirectoryLoader("data").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create FAISS database and retriever
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Create retriever tool
tool = create_retriever_tool(
    retriever,
    "search_health_cancer",
    "Searches and returns Conditions and Surgeries covered in Life Heart Shield Plan",
)
tools = [tool]

# Load the prompt
prompt = hub.pull("hwchase17/openai-tools-agent")

# Create the agent and agent executor
agent = create_openai_tools_agent(llm_azure, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Invoke the agent with the input query
result = agent_executor.invoke(
    {
        "input": "What are the different Conditions and Surgeries covered in Life Heart Shield Plan?"
    }
)

print(result["output"])

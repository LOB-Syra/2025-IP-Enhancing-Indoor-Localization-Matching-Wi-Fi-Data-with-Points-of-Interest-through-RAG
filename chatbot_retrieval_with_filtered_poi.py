import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.document_loaders import JSONLoader
from langchain import hub
# from langchain.chains import create_retrieval_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document 
# from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader



import openai
openai.api_type = "azure"
openai.api_version = "2024-06-01"
azure_endpoint = "https://hkust.azure-api.net"


load_dotenv(dotenv_path='.env')

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

csv_base_dir = "data_filteredpoi"

# 创建 JSON 文本分割器实例
text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

# 初始化文档列表
docs = []

# 遍历 data 文件夹下的 JSON 文件
for item in os.listdir(csv_base_dir):
    if item.endswith('.csv'):
        if item == '.DS_Store':
            continue
        file_path = os.path.join(csv_base_dir, item)
        loader = CSVLoader(
        file_path=file_path,
        csv_args={
            'delimiter': ',',  # 指定分隔符为逗号
            'fieldnames': ['id','types','formattedAddress','latitude','longitude','businessStatus','displayName','primaryTypeDisplayName','primaryType','shortFormattedAddress','gptFloor','gptBuilding']  # 指定列名
        }
    )
        data = loader.load()
        docs.extend(data)

print(len(docs))


# 准备嵌入引擎
embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_version="2024-06-01",
    azure_endpoint=azure_endpoint,  
    openai_api_type="azure")

# 向量化
vectordb = Chroma(embedding_function=embeddings, persist_directory="./vectordb_filteredpoi")
vectordb.add_documents(docs)  



llm = AzureChatOpenAI(
    openai_api_version="2024-06-01",
    deployment_name="gpt-4o",
    model_name="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    openai_api_type="azure",
    openai_api_base=azure_endpoint,  
    temperature=0
)

retriever = vectordb.as_retriever(search_kwargs={"k": 150})


chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)


prompt_template = """
### Context
- I want you to act as a specialist to recognize the most possible 5 shop names from the POI data in your knowledge base for the given AP (Wi-Fi access point) name.
- For example, you can recognize "CTF FREE WIFI" as the AP name of Chow Tai Fook.
- Some access points adopt the name of the brand or business they are associated with.
- In certain cases, access points might be identified using acronyms or abbreviations related to the business or location. Note that the SSID may only contain a part of the shop name or an abbreviation.

### AP
{user_input}

### Task
- Retrieve the 5 most similar POI entries from the database based on the provided AP name.
- For each entry, calculate and provide a similarity score indicating how closely the AP name matches the POI's name or associated brand.
- Return the results in the format specified below.

### Format
- POI Name: [POI Name]
  - Similarity Score: [Score]
  - Details: [Formatted Address, Business Status, Display Name, Primary Type Display Name]
"""

# 循环接收用户输入并进行问答
while True:
    user_input = input("请输入AP名称或相关内容：")  
    full_prompt = prompt_template.format(user_input=user_input)  # 拼接完整 Prompt
    result = chain({"query": full_prompt})
    print(result["result"])

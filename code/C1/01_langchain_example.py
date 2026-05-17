import os
# hugging face镜像设置，如果国内环境无法使用启用该设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

markdown_path = "../../data/C1/markdown/easy-rl-chapter1.md"

# 加载本地markdown文件
loader = UnstructuredMarkdownLoader(markdown_path)
docs = loader.load()

# 文本分块
text_splitter = RecursiveCharacterTextSplitter()
chunks = text_splitter.split_documents(docs)

# 使用HuggingFaceEmbeddings加载之前在初始化设置中下载的中文嵌入模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
  
# 构建向量存储，从而一个可供查询的知识索引
vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(chunks)

# 提示词模板
prompt = ChatPromptTemplate.from_template("""请根据下面提供的上下文信息来回答问题。
请确保你的回答完全基于这些上下文。
如果上下文中没有足够的信息来回答问题，请直接告知：“抱歉，我无法根据提供的上下文找到相关信息来回答此问题。”

上下文:
{context}

问题: {question}

回答:"""
                                          )

# 配置大语言模型

# 使用 AIHubmix
llm = ChatOpenAI(
    model="glm-4.7-flash-free",
    temperature=0.7,
    max_tokens=4096,
    api_key=os.getenv("AIHUBMIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)

# llm = ChatOpenAI(
#     model="deepseek-chat",
#     temperature=0.7,
#     max_tokens=4096,
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     base_url="https://api.deepseek.com"
# )

# 用户查询
question = "文中举了哪些例子？"

# 在向量存储中查询相关文档
# 根据用户问题在索引中查找最相关的 k (此处示例中 k=3) 个文本块
retrieved_docs = vectorstore.similarity_search(question, k=3)
# 准备上下文: 将检索到的多个文本块的页面内容 (doc.page_content) 合并成一个单一的字符串
# 并使用双换行符 ("\n\n") 分隔各个块，形成最终的上下文信息 (docs_content) 供大语言模型参考
docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

# 将检索到的上下文与用户问题结合，利用大语言模型（LLM）生成答案
answer = llm.invoke(prompt.format(question=question, context=docs_content))
print(answer)

import os
from operator import itemgetter

from langchain_chroma import Chroma
from langchain_community.document_loaders.text import TextLoader
from langchain_core.messages.chat import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_dir = os.path.join(os.getcwd(), 'app/docs')
prompts_dir = os.path.join(os.getcwd(), 'app/prompts')

# Get knownlwdge data
knownledge = TextLoader(os.path.join(docs_dir, 'text.txt')).load()[0].page_content

# Split knownledge data
texts = RecursiveCharacterTextSplitter().split_text(knownledge)

# Init Embedding function
embeddings = OllamaEmbeddings(
    base_url="http://192.168.31.91:11434", 
    model="nomic-embed-text"
)

# Create verctor store.
vector_store = Chroma(
    collection_name="reAct_doc", 
    embedding_function=embeddings,
)

# Add text to vector store.
vector_store.add_texts(texts=texts)

# Get retriever function from vector store
retriever = vector_store.as_retriever(
    search_kwargs={'k': 1}
)

# llm instance
llm = ChatOllama(base_url="http://192.168.31.91:11434", model="deepseek-r1:7b")

# Prompt loader
loader = TextLoader(
    os.path.join(prompts_dir, 'AI_assistant.md')
)

# Prompt instance
prompt = PromptTemplate.from_template(
    loader.load()[0].page_content,
)

# Output parser
parser = StrOutputParser()

# llm chain
llm_chain = {
    "context": itemgetter("question") | retriever,
    "question": itemgetter("question")
} | prompt | llm | parser

# Terminal interactive
while True:
    question = input("> ")

    if question.strip() != "":
        quiz = ChatMessage(role="user", content=question.strip())
        answer =llm_chain.invoke({
            "question": question
        })

        print(answer)

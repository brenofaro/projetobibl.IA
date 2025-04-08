import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough

# Import para chat e embeddings da OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

# Carregar variáveis de ambiente
_ = load_dotenv(find_dotenv())

OPENAI_MODEL_NAME = "gpt-4o-mini"

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", None)

# Instancia o modelo da OpenAI
model_local = ChatOpenAI(
    model_name=OPENAI_MODEL_NAME,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=OPENAI_API_BASE,
    streaming=True,
    temperature=0.7
)

# Função de carregamento e indexação dos dados do CSV
# Utilizando o CSV (ex.: bible_questions.csv)
@st.cache_resource
def load_csv_data():
    # Carrega o conjunto de dados
    loader = CSVLoader(file_path="bible_questions.csv")
    documents = loader.load()

    # Gerar embeddings a partir do modelo da OpenAI
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=OPENAI_API_BASE
    )

    # Cria e indexa o vetor usando FAISS
    vectorstore = FAISS.from_documents(documents, embeddings)
    # Cria o objeto de recuperação que busca os documentos relevantes
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    return retriever

# Obtém o retriever com os dados indexados
retriever = load_csv_data()

st.title("Oráculo Bíblico - Responda suas dúvidas")

# Template do prompt
rag_template = """
Você é um assistente que responde perguntas sobre a Bíblia. 
Você deve responder com base no contexto fornecido, e baseado na base de dados fornecida para você. 
Se não houver informações suficientes, diga que não sabe.

Contexto: {context}
Pergunta: {question}
"""
prompt = ChatPromptTemplate.from_template(rag_template)

# Cria o pipeline que recebe o prompt formatado e invoca o modelo de linguagem.
chain = prompt | model_local

# Inicializa o histórico de mensagens no session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe o histórico de mensagens (componente de chat)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Caixa de entrada para o usuário
if user_input := st.chat_input("Você:"):
    # Salva e exibe a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # --- Etapa de recuperação de contexto ---
    # Recupera os documentos relevantes com base na pergunta do usuário.
    docs = retriever.invoke(user_input)
    # Junta o conteúdo dos documentos relevantes em um único texto
    contexto = " ".join([doc.page_content for doc in docs])

    # Executa o pipeline (chain) passando o contexto e a pergunta.
    # Para possibilitar o streaming, utilizamos chain.stream.
    resposta_stream = chain.stream({"context": contexto, "question": user_input})
    full_response = ""

    # Container para a resposta do assistente
    response_container = st.chat_message("assistant")
    response_text = response_container.empty()

    # Exibe a resposta conforme ela é gerada (streaming)
    for partial_response in resposta_stream:
        full_response += str(partial_response.content)
        response_text.markdown(full_response + "▌")

    # Salva a resposta completa no histórico
    st.session_state.messages.append({"role": "assistant", "content": full_response})

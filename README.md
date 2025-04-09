# Projeto Bibl.IA para a disciplina de Inteligência Artificial na UFS

Integrantes do grupo:  
BRENNO DE FARO VIEIRA  
DAVI SOUZA FONTES SANTOS  
HUMBERTO DA CONCEIÇÃO JÚNIOR  
RAFAEL NASCIMENTO ANDRADE  
NEWTON SOUZA SANTANA JÚNIOR  

### Site do Projeto para testes

[Bibl.IA](https://projetobibl-ia.onrender.com/)

### Requisitos para rodar o projeto
- Python 3.8 ou superior
- [Ollama](https://ollama.com/download)


### Comandos
- Criar o venv:
```bash
python -m venv venv
```
- Ativar o venv:
```bash
.\venv\Scripts\activate
```

- Instalar as dependências de python do projeto:
```bash
pip install -r requirements.txt
```
- Baixar a imagem do modelo nomic-embed-text no ollama:
```bash
ollama pull nomic-embed-text
```
- Baixar a imagem do llama3.1:
```bash
ollama pull llama3.1:8b-instruct-q4_K_S
```
- Rodar o servidor:
```bash
streamlit run .\biblIA.py
```

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv, find_dotenv

# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        st.error('Document format is not supported!')
        return None

    return loader.load()

# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

# create embeddings and store in Chroma
def create_embeddings(chunks, persist_directory="./chroma_storage"):
    from chromadb.config import Settings
    client = chromadb.PersistentClient(path=persist_directory)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    collection_name = "my_collection"
    collection = client.get_or_create_collection(collection_name)

    ids = [str(i) for i in range(len(chunks))]
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings.embed_documents(texts)
    )

    return Chroma(client=client, collection_name=collection_name, embedding_function=embeddings)

# retrieve and answer questions
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.invoke(q)
    return answer['result']

# calculate embedding cost
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    return total_tokens, total_tokens / 1000 * 0.00002

# clear chat history
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)

    st.subheader('LLM Question-Answering Application 🤖')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, chunking and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                if data is None:
                    st.error("Failed to load document.")
                else:
                    chunks = chunk_data(data, chunk_size=chunk_size)
                    st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                    tokens, embedding_cost = calculate_embedding_cost(chunks)
                    st.write(f'Embedding cost: ${embedding_cost:.4f}')

                    vector_store = create_embeddings(chunks)
                    st.session_state.vs = vector_store
                    st.success('File uploaded, chunked, and embedded successfully.')

    q = st.text_input('Ask a question about the content of your file:')
    if q:
        standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                          # "If you can't answer then return `I DONT KNOW`."
        q = f"{q} {standard_answer}"

        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            st.text_area('LLM Answer: ', value=answer, height=300)

            st.divider()

            if 'history' not in st.session_state:
                st.session_state.history = ''

            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            st.text_area(label='Chat History', value=h, key='history', height=400)

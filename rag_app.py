from langchain_openai import ChatOpenAI
import streamlit as st


def load_document(file):
    '''
    Load file(s)
    '''
    import os
    _, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        raise ValueError("Document format is not supported")

    data_from_file = loader.load()

    return data_from_file


def chunk_data(data,
               chunk_size=1000,
               chunk_overlap=200):
    '''
    Split text in chunks
    '''
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)

    return chunks


def create_embeddings_chroma(chunks,
                             model_name='text-embedding-3-large',
                             persist_dir='./chroma_db'):
    '''
    Make embeddings and use Chromadb as vector store
    '''
    # from langchain_chroma import Chroma
    import os
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    embedding_function = OpenAIEmbeddings(model=model_name)
    vector_store = Chroma.from_documents(chunks,
                                         embedding_function,
                                         collection_name="vectors",
                                         persist_directory=persist_dir)
    return vector_store


def ask_and_get_answer(vector_store,
                       model_name,
                       q,
                       k_nn):
    '''
    Asking and getting questions
    '''
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    retriever = vector_store.as_retriever(search_type='similarity',
                                          search_kwargs={'k': k_nn})
    chat = ChatOpenAI(model_name=model_name,
                      temperature=0)

    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(chat,
                                                         prompt)
    chain = create_retrieval_chain(retriever,
                                   question_answer_chain)
    answer = chain.invoke({"input": q})

    return answer['answer']


def calculate_embedding_cost(texts, model_name):
    '''
    Calculate the OpenAI embedding costs
    '''
    import tiktoken
    enc = tiktoken.encoding_for_model(model_name)
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    return (total_tokens, 0.0004 * total_tokens / 1000)


def clear_history():
    '''
    Clear history callback
    '''
    if 'history' in st.session_state:
        del st.session_state['history']


if __name__ == '__main__':
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)

    LLM = "gpt-3.5-turbo" #'gpt-4o-2024-08-06'  # 
    EMBEDDING_MODEL = 'text-embedding-3-large'
    CHROMA_PATH = './chroma_db'

    # st.image('img.png')
    st.subheader('LLM QA app')

    with st.sidebar:
        api_key = st.text_input('OpenAI API Key:', type='password')
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        uploaded_file = st.file_uploader('Upload a file: ',
                                         type=['pdf', 'docx', 'txt'])
        chunk_size = st.number_input('Chunk size: ',
                                     min_value=100,
                                     max_value=2048,
                                     value=512,
                                     on_change=clear_history)
        chunk_overlap = st.number_input('Chunk_overlap: ',
                                     min_value=0,
                                     max_value=500,
                                     value=250,
                                     on_change=clear_history)

        k_nn = st.number_input('k',
                               min_value=1,
                               max_value=5,
                               value=3,
                               on_change=clear_history)
        add_data = st.button('Add data',
                             on_click=clear_history)

        if uploaded_file and add_data:
            with st.spinner('Reading, splitting and embedding file ...'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)

                if chunk_overlap > chunk_size:
                    chunk_overlap = round(chunk_size / 5)

                chunks = chunk_data(data,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap)

                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks,
                                                                  model_name=EMBEDDING_MODEL)
                st.write(f'Embedding_cost: {embedding_cost:4f}')

                vector_store = create_embeddings_chroma(chunks,
                                                        model_name=EMBEDDING_MODEL,
                                                        persist_dir=CHROMA_PATH)
                st.session_state.vs = vector_store
                st.success('File uploaded, chunked and embedded successfully!')

    q = st.text_input('Ask a question about the file content')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            answer = ask_and_get_answer(vector_store=vector_store,
                                        model_name=LLM,
                                        q=q,
                                        k_nn=k_nn)
            st.text_area('LLM answer: ',
                         value=answer)

    st.divider()
    if 'history' not in st.session_state:
        st.session_state.history = ''
    value = f'Q: {q} n\A: {answer}'
    st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
    h = st.session_state.history
    st.text_area(label='Chat History',
                 value=h,
                 key='history',
                 height=400)
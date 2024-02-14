from  langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings  import  HuggingFaceEmbeddings

def load_pdf(data):
    loader = DirectoryLoader(data,
                             glob = "*.pdf",
                             loader_cls = PyPDFLoader)
    
    document = loader.load()
    
    return document

#Creating Text Chunks
def text_split(extended_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extended_data)
    
    return text_chunks


def download_huggingface_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
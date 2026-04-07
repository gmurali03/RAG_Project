# rag_app.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


# 1. Load PDF
pdf_path = "C:/Users/mural/OneDrive/Desktop/G Nandini.pdf"

print("Loading PDF...")
loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"Loaded {len(documents)} pages")


# 2. Split Text
print("\nSplitting text...")
text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = text_splitter.split_documents(documents)
print(f"Created {len(docs)} chunks")


# 3. Create Embeddings
print("\nCreating embeddings (first time may take time)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# 4. Create Vector Database
vector_db = FAISS.from_documents(docs, embeddings)
print("Vector DB ready!")


# 5. Ask Questions (Interactive)
print("\n💬 Ask questions from your PDF (type 'exit' to quit)\n")

while True:
    query = input("👉 Your question: ")

    if query.lower() == "exit":
        break

    similar_docs = vector_db.similarity_search(query)

    context = " ".join([doc.page_content for doc in similar_docs])

    print("\n📄 Answer:\n")
    print(context[:1000])
    print("\n" + "-"*50)
    
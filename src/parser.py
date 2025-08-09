import os
import git
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def is_text_file(filepath, blocksize=512):
    """
    Check if a file is likely text by reading a small block.
    """
    try:
        with open(filepath, "rb") as f:
            chunk = f.read(blocksize)
        if b"\x00" in chunk:
            return False
        try:
            chunk.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False
    except Exception:
        return False

def fetch_repo(repo_url, branch_or_commit, local_path="repo_temp"):
    """
    Clones or fetches a git repository and checks out a specific branch, commit, or tag.
    """
    if os.path.exists(local_path):
        print(f"Using existing repo at {local_path}")
        repo = git.Repo(local_path)
        repo.remotes.origin.fetch()
    else:
        print(f"Cloning {repo_url} into {local_path}...")
        repo = git.Repo.clone_from(repo_url, local_path)

    print(f"Checking out {branch_or_commit}...")
    repo.git.checkout(branch_or_commit)
    return local_path

def load_text_documents(repo_path):
    """
    Loads only text-like files from the repo by content, skipping binaries.
    """
    docs = []
    for filepath in Path(repo_path).rglob("*"):
        if filepath.is_file() and is_text_file(filepath):
            try:
                loader = TextLoader(str(filepath), encoding="utf-8")
                docs.extend(loader.load())
            except Exception as e:
                print(f"Skipping {filepath}: {e}")
    return docs


def prepare_codebase_for_vectors(repo_path, output_path="vectorstore"):
    """
    Convert a codebase into vector embeddings for use with LangChain/LangFlow/CrewAI.
    Local embeddings only, no API keys required.
    """
    documents = load_text_documents(repo_path)
    print(f"Loaded {len(documents)} text-like files from {repo_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(output_path)
    print(f"Vector store saved to {output_path}")

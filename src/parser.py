import os
import git
from git.exc import GitCommandError
from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EXCLUDE_DIRS = {".git", "node_modules", "venv", "dist", "build", "__pycache__"}

def is_text_file(filepath, blocksize=512):
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

def fetch_repo(repo_url, branch_or_commit, local_path="repo_temp", github_token=None):
    def is_commit_hash(s):
        return all(c in "0123456789abcdef" for c in s.lower()) and len(s) in {7, 40}

    if github_token and repo_url.startswith("https://"):
        repo_url = repo_url.replace("https://", f"https://{github_token}@")

    try:
        if os.path.exists(local_path):
            print(f"Using existing repo at {local_path}")
            repo = git.Repo(local_path)
            repo.remotes.origin.fetch()
        else:
            if is_commit_hash(branch_or_commit):
                print(f"Cloning full history to get commit {branch_or_commit}...")
                repo = git.Repo.clone_from(repo_url, local_path)
            else:
                print(f"Shallow cloning {branch_or_commit} from {repo_url}...")
                repo = git.Repo.clone_from(
                    repo_url,
                    local_path,
                    branch=branch_or_commit,
                    depth=1,
                    single_branch=True
                )

        print(f"Checking out {branch_or_commit}...")
        repo.git.checkout(branch_or_commit)
        return local_path

    except GitCommandError as e:
        if "Authentication failed" in str(e):
            raise RuntimeError(
                "Authentication failed. If this is a private repo, provide a valid GitHub token."
            )
        else:
            raise

def load_text_documents(repo_path):
    docs = []
    for filepath in Path(repo_path).rglob("*"):
        if not filepath.is_file():
            continue

        if any(part in EXCLUDE_DIRS for part in filepath.parts):
            continue

        if not is_text_file(filepath):
            continue

        try:
            loader = TextLoader(str(filepath), encoding="utf-8")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Skipping {filepath}: {e}")
    return docs

def prepare_codebase_for_vectors(repo_path, output_path="vectorstore"):
    documents = load_text_documents(repo_path)
    print(f"Loaded {len(documents)} text-like files from {repo_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"},
        encode_kwargs={"batch_size": 64}
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(output_path)
    print(f"Vector store saved to {output_path}")

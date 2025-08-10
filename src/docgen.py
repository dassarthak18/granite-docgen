import os, shutil
import torch
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from transformers import pipeline
from parser import fetch_repo, prepare_codebase_for_vectors

REPO_URL = os.environ.get("REPO_URL")
if not REPO_URL:
    raise RuntimeError("REPO_URL environment variable not set. Please specify the git repository.")
BRANCH_OR_COMMIT = os.environ.get("BRANCH_OR_COMMIT")
if not BRANCH_OR_COMMIT:
    raise RuntimeError("BRANCH_OR_COMMIT environment variable not set. Please specify the branch or commit to target.")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    LOCAL_REPO_PATH = fetch_repo(REPO_URL, BRANCH_OR_COMMIT)
else:
    LOCAL_REPO_PATH = fetch_repo(REPO_URL, BRANCH_OR_COMMIT, github_token=GITHUB_TOKEN)

prepare_codebase_for_vectors(LOCAL_REPO_PATH)
shutil.rmtree(LOCAL_REPO_PATH)

embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cuda" if os.environ.get("USE_GPU", "0") == "1" else "cpu"},
        encode_kwargs={"batch_size": 64}
)

vectorstore = FAISS.load_local("vectorstore", embeddings=embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

model_path = "./granite"
device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    device=device,
    max_length=2048,
    do_sample=True,
    temperature=0.8,
    top_p=0.8,
)

llm = HuggingFacePipeline(pipeline=pipe)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

with open("query.txt", "r", encoding="utf-8") as f:
    query = f.read().strip()
outputs = qa_chain.invoke({"query": query})
print("Generated Documentation:\n", outputs["result"])

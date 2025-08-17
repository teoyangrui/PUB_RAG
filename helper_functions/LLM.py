import os
import re
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from openai import OpenAI

from langchain.schema.retriever import BaseRetriever
from typing import Optional, List
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.prompts import PromptTemplate

load_dotenv(Path(__file__).resolve().parent / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")
with open("./helper_functions/label_map.json", "r", encoding="utf-8") as f:
    global_label_map = json.load(f)

#load vectordb
embedding = OpenAIEmbeddings(openai_api_key=API_KEY)
vectordb = Chroma(
    persist_directory="chroma_parent_child",
    embedding_function=embedding
)


### This part is to extract the annex => map using mapping json => retrieve annex data from metadata
# references in documents come in many different forms and need to be normalized. e.g. annex K, AnnexK, Annex  K -> annex k
ref_regex = re.compile(
    r"\b(?:annex|appendix)\s+[a-z\d]+\b"
    r"|\bdrawing\s+no\.?\s*\d+\b"
    r"|\bfigure\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE
)

def normalize_label(label: str) -> str: 
    label = label.lower().replace("\n", " ")
    label = re.sub(r"\s+", " ", label.strip())
    label = re.sub(r"(\bno)\s*(\d)", r"\1 \2", label)  
    return label
def extract_annex_refs(text):
    matches = ref_regex.findall(text)
    return list(set([normalize_label(m) for m in matches]))

def fetch_annex_boost(vectordb: Chroma, query_refs: list[str]) -> list[Document]:
    if not query_refs:
        return []

    # Build a metadata filter:
    # - annex_refs is a LIST field -> use $contains once per ref
    # - figure_label_norm is a SCALAR -> use $in
    where = {
        "$or": (
            [{"annex_refs": {"$contains": r}} for r in query_refs] +
            [{"figure_label_norm": {"$in": query_refs}}]
        )
    }

    # Access the underlying Chroma collection to avoid embeddings/query. fetch the annex references mapped using label
    res = vectordb._collection.get(               
        where=where,
        include=["documents", "metadatas"]
    )

    return [
        Document(page_content=d, metadata=m)
        for d, m in zip(res["documents"], res["metadatas"])
    ]

class AnnexAwareRetriever(BaseRetriever):
    def __init__(self, base_retriever, label_map, query=""):
        super().__init__()
        self._base_retriever = base_retriever
        self._label_map = label_map
        self._query = query

    def get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        query_refs = extract_annex_refs(query)
        mapped_refs = [self._label_map.get(r, r) for r in query_refs]
        base_results = self._base_retriever.get_relevant_documents(query)

        # Boost docs with matching annex refs
        annex_boost = fetch_annex_boost(vectordb, query_refs)
        return annex_boost + base_results

#define prompts
RAG_ROLE = (
    '''
    You are a helpful expert Public Officer from Singapore's Public Utilities Board.
    Answer using ONLY the provided context from PUB Codes of Practice unless noted otherwise. 
    Do not invent facts. Be concise and include important specifications. 
    If a question is outside your scope, state this upfront and advise verification.
    '''
)

RAG_INSTRUCTIONS = (
    '''
    Use ONLY the context below to answer.
    If the answer is not present in the context, reply exactly: 'Not found in the uploaded documents.' 
    When you cite, use the format [source: document_name p.page].
    '''
)

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        f"{RAG_ROLE}\n\n"
        f"{RAG_INSTRUCTIONS}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


#create QA chain
#mmr => multi query => load my annexaware retriever
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
base_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.6})
multi_query_retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)
retriever = AnnexAwareRetriever(base_retriever=multi_query_retriever, label_map=global_label_map)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    return_source_documents=True,
    chain_type_kwargs = {"prompt": RAG_PROMPT})

#Ask Function
def ask(query, temp_context=False):
    #using persistant vectorDB
    if temp_context == False:

        result = qa_chain({"query": query})
        answer = result["result"]
    elif temp_context: #only when using temp context. 
        # temp_context=True, use OpenAI directly.
        # Expect `query` to already contain any context block you assembled upstream (e.g., "Context excerpts: ...").
        client = OpenAI(api_key=API_KEY)
        system_msg = (
            '''
            You are a concise, accurate assistant.
            "If the user's message includes a 'Context' or 'Context excerpts' block, 
            ground your answer strictly in it and cite like [source: p.page]. 
            If no context is present, answer concisely from general knowledge.
            '''
        )
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
        )
        answer = resp.choices[0].message.content
    

    return answer #+"\n\n\n" + "\n\n".join(to_add)
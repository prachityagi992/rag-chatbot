#!/usr/bin/env python
# coding: utf-8

# ### Data Ingestion
# 

# In[14]:


###Document Structure

from langchain_core.documents import Document


# In[40]:


doc=Document(
    page_content="this is the main text content I am using to create RAG",
    metadata={
        "source":"example.txt",
        "pages":1,
        "author":"prachi tyagi",
        "date_created":"12-06-2026"
    }
)
doc


# In[41]:


## created a simple txt file
import os
os.makedirs("../data/text_files",exist_ok=True)


# In[42]:


sample_texts={
    "../data/text_files/python_intro.txt":"Python Programming Introduction"
"""Python is a high-level, interpreted, and general-purpose programming language known for its simple syntax and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms, including object-oriented, procedural, and functional programming.

Python is widely used in web development, data science, artificial intelligence (AI), machine learning, automation, and software development. Its extensive collection of libraries and frameworks makes development faster and more efficient.

For Retrieval-Augmented Generation (RAG) applications, Python is a popular choice because it provides powerful libraries such as LangChain, FAISS, ChromaDB, Transformers, and OpenAI SDKs. These tools help developers build intelligent systems that can retrieve relevant information from documents and generate accurate, context-aware responses.

Key Features of Python:

Easy to learn and use
Simple and readable syntax
Cross-platform compatibility
Large standard library
Strong community support
Excellent support for AI and Machine Learning""",

"../data/text_files/machine_learning.txt": """Machine Learning Basics
 Machine Learning (ML) is a branch of Artificial Intelligence that enables
 computers to learn from data and improve their performance without being explicitly programmed for every task. Instead of following fixed rules, machine learning algorithms identify patterns in data and use those patterns to make predictions or decisions.

Machine learning is widely used in applications such as recommendation systems, image recognition, speech recognition, fraud detection, healthcare, and chatbots. By analyzing large amounts of data, ML models can continuously improve their accuracy over time.

Types of Machine Learning
Supervised Learning : Learns from labeled data to make predictions.
Unsupervised Learning : Finds hidden patterns in unlabeled data.
Reinforcement Learning : Learns through trial and error using rewards and penalties.
Key Features
Learns from data automatically
Improves performance with experience
Handles large datasets efficiently
Identifies complex patterns and relationships
Supports predictive analytics and automation
Applications of Machine Learning
Recommendation systems (Netflix, YouTube, Amazon)
Email spam detection
Face and image recognition
Voice assistants
Medical diagnosis
Financial forecasting
Autonomous vehicles"""


}
for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8")as f:
        f.write(content)
        print("sample text files created")



# In[55]:


### TextLoader

from langchain_community.document_loaders import TextLoader
loader=TextLoader("../data/text_files/python_intro.txt",encoding="utf-8")
document=loader.load()
print(document)


# In[77]:


### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt", ##pattern to match files
loader_cls=TextLoader, ##loader class to use
loader_kwargs={'encoding':'utf-8'},
show_progress=False
)
documents=dir_loader.load()
documents



# In[78]:


from langchain_community.document_loaders import  PyMuPDFLoader

dir_loader = DirectoryLoader(
    "data/pdf",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader,
    show_progress=True
)

pdf_documents = dir_loader.load()

print("Documents:", len(pdf_documents))


# In[79]:


type(pdf_documents[0])


# ### Chunking

# In[80]:


from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=200
):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example chunk
    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs


# Function call
chunks = split_documents(pdf_documents)


# In[81]:


from langchain_text_splitters import RecursiveCharacterTextSplitter


# ### Embedding and Vector Store DataBase
# 

# In[82]:


import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List , Dict , Any ,Tuple
from sklearn.metrics.pairwise import cosine_similarity


# In[62]:


class EmbeddingManager:
    """Handle document embedding generation using SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager

        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()


    def _load_model(self):
        """Load the Sentence Transformer"""
        try:
            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(self.model_name)

            print(
                f"Model Loaded Successfully. Embedding dimension: "
                f"{self.model.get_sentence_embedding_dimension()}"
            )

        except Exception as e:
            print(f"Error Loading model {self.model_name}: {e}")
            raise


    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")

        print(f"Generating embeddings for {len(texts)} texts...")

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings


# Initialize the embedding manager (class ke bahar)

embedding_manager = EmbeddingManager()

embedding_manager


# In[63]:


from sentence_transformers import SentenceTransformer


# ### Vector Store

# In[64]:


import os
import uuid
import numpy as np
import chromadb
from typing import List, Any


# In[65]:


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""

    def __init__(
        self,
        collection_name: str = "pdf_documents",
        persist_directory: str = "../data/vector_store"
    ):
        """
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()



    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)

            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "PDF document embeddings for RAG"
                }
            )

            print(
                f"Vector store initialized. Collection: {self.collection_name}"
            )

            print(
                f"Existing documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise




    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray
    ):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """

        if len(documents) != len(embeddings):
            raise ValueError(
                "Number of documents must match number of embeddings"
            )

        print(
            f"Adding {len(documents)} documents to vector store..."
        )

        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(
            zip(documents, embeddings)
        ):

            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(
                embedding.tolist()
            )

        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(
                f"Successfully added {len(documents)} documents "
                f"to vector store"
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(
                f"Error adding documents to vector store: {e}"
            )
            raise


# Create vector store object

vectorstore = VectorStore()

vectorstore


# In[66]:


chunks


# ### Core Flow Of Rag Pipeline

# In[67]:


### Convert the text to embeddings
texts = [doc.page_content for doc in chunks]

## Generate the Embeddings
embeddings = embedding_manager.generate_embeddings(texts)

## Store in the vector database
vectorstore.add_documents(chunks, embeddings)


# ### Retriever Pipeline From VectorStore

# In[ ]:


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results["documents"] and results["documents"][0]:

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]
                distances = results["distances"][0]
                ids = results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):

                    similarity_score = 1 - distance

                    if distance <= 2.0:

                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": document,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1
                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

            else:
                print("No documents found")

            return retrieved_docs


        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []



vector_store = VectorStore()
embedding_manager = EmbeddingManager()

rag_retriever = RAGRetriever(vector_store, embedding_manager)


# In[69]:


rag_retriever


# In[70]:


rag_retriever.retrieve("what is attention is all you need")


# In[71]:


### get the context from the retriever and pass it to the LLM

rag_retriever.retrieve("Unified Multi-task Learning Freamework")


# ### Integration Vectordb Context pipeline With LLM  output

# In[72]:


from langchain_groq import ChatGroq
import os ##to access environment variable(.env)
from dotenv import load_dotenv ## to load (.env)

# Load environment variables
load_dotenv(r"c:\Users\prakr\OneDrive\Desktop\RAG--\notebooks\.env")

# Get API key from(.env)
groq_api_key = os.getenv("GROQ_API_KEY")

# Check API key found or not
print("API Key Loaded:", bool(groq_api_key))

# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.1,##cntrl creativity
    max_tokens=1024 ##reponse maxx length
)

# Simple RAG function : retrieve context + generate response
def rag_simple(query, retriever, llm, top_k=3):

    # Step 1: Retrieve context
    results = retriever.retrieve(query, top_k=top_k) ##vector db m search krega

    # Step 2: Build context safely
    context = "\n\n".join( ##this combine the chunks
        [
            doc["content"] if isinstance(doc, dict) else doc.page_content
            for doc in results
        ]
    ) if results else ""

    # If no context found
    if not context.strip():
        return "No relevant context found to answer the question."

    # Step 3: generate the answer using GROQ LLM
    prompt = f"""
Use the following context to answer the question concisely and correctly.

## pass the data to llm from retriever

Context:
{context}

Question:
{query}

Answer:
"""

    # Step 4: LLM response
    response = llm.invoke([prompt.format(context, query=query)])
    return response.content




# In[73]:


rag_retriever = RAGRetriever(vector_store, embedding_manager)


# In[74]:


answer=rag_simple("what is cybercrime", rag_retriever,llm)
print(answer)


# ### Enhanced RAG pipeline Features
# 

# In[83]:


# --- Enhanced RAG Pipeline Features ---  
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):  
  """  
  RAG pipeline with extra features:  
  - Returns answer, sources, confidence score, and optionally full context.  
  """  
  results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)  

  if not results:  
    return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}  

  # Prepare context and sources  
  context = "\n\n".join([doc['content'] for doc in results])  

  sources = [{  
      'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),  
      'page': doc['metadata'].get('page', 'unknown'),  
      'score': doc['similarity_score'],  
      'preview': doc['content'][:120] + '...'  
  } for doc in results]  

  confidence = max([doc['similarity_score'] for doc in results])  ##find the max similarity score

  ##Generate prompt
  prompt = f"Use the following context to answer the question concisely.\nContext :\n{context}\nn\nQuestion:{query}\n\nAnswer"

  response = llm.invoke([prompt.format(context=context,query=query)])

  output = {  
    'answer': response.content,  
    'sources': sources,  
    'confidence': confidence  
  }  

  if return_context:  
    output['context'] = context  

  return output  

# Function call + To Show Output:  

result = rag_advanced("What is attention mechanism?", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)  

print("Answer:", result['answer'])  
print("Sources:", result['sources'])  
print("Confidence:", result['confidence'])  
print("Context Preview:", result['context'][:300])


# ### Advanced Rag Pipeline

# In[ ]:


# --- Advanced RAG Pipeline: 
# Streaming===  we got the answer token by token
# Citations === give document name also in which ans exist
# History== store user previous questions
# Summarization=== convert long documnets into short 

from typing import List, Dict, Any ##typing== data type btane ke liye
import time ## to delay the streaming 

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query n ans

       ##ACtual rag run...
    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        stream: bool = False,
        summarize: bool = False
    ):

        # Retrieve relevant documents
        results = self.retriever.retrieve(
            question,
            top_k=top_k,
            score_threshold=min_score
        )

        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""

        else:
            context = "\n\n".join(
                [doc["content"] for doc in results]
            )

            sources = [
                {
                    "source": doc["metadata"].get(
                        "source_file",
                        doc["metadata"].get("source", "unknown")
                    ),
                    "page": doc["metadata"].get("page", "unknown"),
                    "score": doc["similarity_score"],
                    "preview": doc["content"][:120] + "..."
                }
                for doc in results
            ]

            # give instruction to llm ,,read ques nd ans it
            prompt = f"""
Use the following context to answer the question concisely.

Context:
{context}

Question:
{question}

Answer:
"""

            if stream: ##divide into chunks ,,without streaming user has to w8 ,,with this they got
                  ##they got instant output
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end="", flush=True)
                    time.sleep(0.05)
                print()

            response = self.llm.invoke(prompt) ##llm is making the ans
            answer = response.content ##ans store

        # Add citations to answer
        citations = [
            f"[{i+1}] {src['source']} (page {src['page']})" ##giving numbering to the source 
            for i, src in enumerate(sources)
        ]

        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations)
            if citations
            else answer
        )

        # Optionally summarize answer
        summary = None

        if summarize and answer:
            summary_prompt = (
                f"Summarize the following answer in 2 sentences:\n{answer}"
            )

            summary_resp = self.llm.invoke(summary_prompt)
            summary = summary_resp.content

        # Store query history
        self.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary
            }
        )
   ##final output in dictionary
        return {
            "question": question,
            "answer": answer_with_citations,

            "sources": sources,
            "summary": summary,
            "history": self.history
        }


# Example usage

adv_rag = AdvancedRAGPipeline(rag_retriever, llm) ##obj bna 

result = adv_rag.query( ##function cll
    "What is attention is all you need?",
    top_k=3,
    min_score=0.3,
    stream=True,
    summarize=True
)

print("\nFinal Answer:", result["answer"])
print("Summary:", result["summary"])
print("History:", result["history"][-1])


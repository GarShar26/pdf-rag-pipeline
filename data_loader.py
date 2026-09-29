import os
from typing import List, Any

import ollama  
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()


EMBED_MODEL = "nomic-embed-text"

node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    reader = PDFReader()
    documents = reader.load_data(file=path)
    nodes = node_parser.get_nodes_from_documents(documents)
    return [{"text": node.text, "source": os.path.basename(path)} for node in nodes]


# data_loader.py

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    all_embeddings = []
    # ⬇️ CHANGE THIS TO 1
    BATCH_SIZE = 1

    print(f"Embedding {len(texts)} chunks (Safe Mode: 1 by 1)...")

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        try:
            response = ollama.embed(
                model=EMBED_MODEL,
                input=batch
            )
            all_embeddings.extend(response["embeddings"])
        except Exception as e:
            # This prints exactly which chunk caused the crash
            print(f"Error on chunk {i}: {e}")
            raise e

    return all_embeddings
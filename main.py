import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import ollama  
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        raw_chunks = load_and_chunk_pdf(pdf_path)
        just_text_chunks = [c["text"] for c in raw_chunks]
        return RAGChunkAndSrc(chunks=just_text_chunks, source_id=source_id)

    def _upsert(data: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = data.chunks
        source_id = data.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    question = ctx.event.data.get("question")
    if not question:
        return {"error": "Missing required field 'question' in event data.", "status": 400}

    top_k = int(ctx.event.data.get("top_k", 5))

    def _search(q_text: str, k: int) -> RAGSearchResult:
        
        query_vec = embed_texts([q_text])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k=k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    if not found.contexts:
        return {"answer": "No relevant information found in the document.", "sources": []}

    context_block = "\n\n".join(f" -{c}" for c in found.contexts)

    
    def _generate_answer():
        
        response = ollama.chat(
            model="llama3",  
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Use the provided context to answer the user's question. If the answer is not in the context, say you don't know."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context_block}\n\nQuestion: {question}"
                }
            ]
        )
        return response['message']['content']

    answer_text = await ctx.step.run("generate-answer", _generate_answer)

    return {"answer": answer_text, "sources": found.sources, "num_contexts": len(found.contexts)}


app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
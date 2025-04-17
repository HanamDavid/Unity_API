from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_output_dir = "audio_responses"
os.makedirs(audio_output_dir, exist_ok=True)

# Loading embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
db_path = "knowledge_base.faiss"  # path for the faiss file
db = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

# PromptTemplate
prompt_template = """You are an AI assistance model. Use the context information to answer the user, if you dont have the answer be smart and try to give him an appropiate answer based on what you know but tell him that you dont really know.

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

@app.post("/process")
async def process(audio: UploadFile):
    # Save audio to temp file
    with open("temp.wav", "wb") as f:
        f.write(await audio.read())

    # STT (Whisper.cpp)
    stt_process = subprocess.run(["./whisper.cpp/build/bin/whisper-cli", "-m", "./whisper.cpp/models/ggml-base.en.bin", "-f", "temp.wav", "-l", "en", "-otxt"], capture_output=True, text=True)
    stt_result = stt_process.stdout.strip()
    print(f"STT Result: {stt_result}")

    # Search in db
    docs = db.similarity_search(stt_result)
    context = "\n".join([doc.page_content for doc in docs])

    # format prompt
    prompt = PROMPT.format(context=context, question=stt_result)

    # LLM (Ollama)
    llm_process = subprocess.run(["ollama", "run", "llama3:8b-instruct-q4_0", prompt], capture_output=True, text=True)
    llm_response = llm_process.stdout.strip()
    print(f"LLM Response: {llm_response}")

    # TTS (Coqui)
    tts_output_path = os.path.join(audio_output_dir, "response.wav")
    tts_process = subprocess.run(["tts", "--text", llm_response, "--out_path", tts_output_path])
    print(f"TTS Audio saved to: {tts_output_path}")

    return {"audio": tts_output_path, "text": llm_response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

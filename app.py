# app.py — LangChain 2025 compatible
import os, asyncio, gradio as gr, edge_tts
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai

load_dotenv()
DB_PATH = "./chroma_pdfs"
VOICE = "en-US-AriaNeural"

# Initialize retriever
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
retriever = Chroma(
    persist_directory=DB_PATH,
    embedding=embeddings,
).as_retriever(search_kwargs={"k": 3})

def answer(question: str) -> str:
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([d.page_content for d in docs])
    prompt = (
        "You are a friendly diabetes assistant. Use only the context below. "
        "Answer in 1-2 sentences. If unsure, say 'I don't know'.\n\n"
        f"Context: {context}\n\nQ: {question}\nA:"
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return response["choices"][0]["message"]["content"].strip()

async def speak(text: str):
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save("reply.mp3")
    os.system("afplay reply.mp3")  # macOS audio player

# --- Gradio Interface ---
with gr.Blocks(title="DiaBot PDF Voice") as demo:
    gr.Markdown("## 💬 Chat with your Diabetes PDF (Voice or Text)")
    with gr.Row():
        audio_in = gr.Audio(source="microphone", type="filepath", label="🎤 Speak")
        text_in  = gr.Textbox(placeholder="Type here…", lines=1)
    send_btn = gr.Button("Send")
    chat = gr.Chatbot(height=400)
    audio_out = gr.Audio(label="Bot Reply", autoplay=True)

    def respond(user_msg, history):
        bot_reply = answer(user_msg)
        history.append((user_msg, bot_reply))
        asyncio.run(speak(bot_reply))
        return history, "reply.mp3"

    send_btn.click(respond, [text_in, chat], [chat, audio_out])
    text_in.submit(respond, [text_in, chat], [chat, audio_out])
    audio_in.stop_recording(
        lambda a, h: respond(
            openai.Audio.transcribe("whisper-1", open(a, "rb"))["text"], h
        ),
        [audio_in, chat],
        [chat, audio_out],
    )

if __name__ == "__main__":
    demo.launch()

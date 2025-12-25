# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Initialize FastAPI
app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input model
class TextInput(BaseModel):
    text: str

# 🔹 Load a small pretrained model once at startup
generator = pipeline("text-generation", model="distilgpt2")  # lighter version of GPT-2

@app.post("/predict")
def predict_next_word(input: TextInput):
    text = input.text.strip()
    if not text:
        return {"prediction": "Please enter some text."}

    # Generate continuation (1-2 new tokens)
    output = generator(text, max_new_tokens=2, num_return_sequences=1)[0]["generated_text"]
    # Extract only the newly generated part
    continuation = output[len(text):].strip()

    return {"prediction": continuation or "(no prediction)"}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# Define model path
MODEL_PATH = "./models/t5-small-radiology-final"

# Detect Device
device = torch.device("cuba" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

app = FastAPI(
    title="Summarising Medical Text - Radiology",
    description="API that summarizes radiology findings into impression text.",
    version="1.0.0"
)

# Request body schema
class FindingsRequest(BaseModel):
    findings:str

#Response body schema
class ImpressionResponse(BaseModel):
    impression:str

@app.get("/")
async def root():
    return {"message": "Radiology Summarisation API is running"}

@app.post("/summarize", response_model=ImpressionResponse)
def summarize_report(request: FindingsRequest):
    if not request.findings or not request.findings.strip():
        raise HTTPException(status_code=400, detail="The 'findings' field is required.")
    
    input_text = "summarize: " + request.findings.strip()
       
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    ouput_ids = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )

    impression = tokenizer.decode(ouput_ids[0], skip_special_tokens=True)

    return ImpressionResponse(impression=impression)
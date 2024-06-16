from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from utils import preprocess_text, load_label_mapping
from starlette.responses import JSONResponse
from functools import lru_cache
import logging

app = FastAPI()

logger = logging.getLogger(__name__)

class Document(BaseModel):
    document_text: str = Field(..., min_length=1, description="The text of the document to classify")

@lru_cache()
def load_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

@lru_cache()
def load_tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased')

CONFIDENCE_THRESHOLD = 0.45

async def classify_document_task(document_text: str, model: BertForSequenceClassification, tokenizer: BertTokenizer):
    document_text = preprocess_text(document_text)
    inputs = tokenizer(document_text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = F.softmax(outputs.logits, dim=1)
        max_scores, preds = torch.max(prediction, dim=1)

    if max_scores.item() < CONFIDENCE_THRESHOLD:
        predicted_label = "other"
    else:
        label_dict = load_label_mapping('label_dict.json')
        predicted_label = label_dict[preds.item()]

    return predicted_label

@app.post("/classify_document")
async def classify_document(document: Document, background_tasks: BackgroundTasks, model: BertForSequenceClassification = Depends(load_model), tokenizer: BertTokenizer = Depends(load_tokenizer)):
    try:
        predicted_label = await classify_document_task(document.document_text, model, tokenizer)
        return {"message": "Classified successfully", "predicted_label": predicted_label}
    except Exception as e:
        logger.exception("An error occurred during document classification")
        raise HTTPException(status_code=500, detail="An error occurred during document classification")

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "The input document is not correctly formatted. Check your JSON: Ensure that the document_text field is of valid length, that the newline characters are escaped (use \\n) and that the JSON is properly formatted.", "details": exc.errors()},
    )
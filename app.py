from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
from finetune_bert_classifier import load_label_mapping
from utils import preprocess_text
from starlette.responses import JSONResponse

app = FastAPI()

class Document(BaseModel):
    document_text: str

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=10)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

CONFIDENCE_THRESHOLD = 0.45

@app.post("/classify_document")
async def classify_document(document: Document):
    if not document.document_text:
        raise HTTPException(status_code=400, detail="The 'document_text' field is required")

    # Tokenize and predict
    try:
        document_text = preprocess_text(document.document_text)
        inputs = tokenizer(document_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = F.softmax(outputs.logits, dim=1)
            max_scores, preds = torch.max(prediction, dim=1)

        # Check if the model is confident enough
        if max_scores.item() < CONFIDENCE_THRESHOLD:
            predicted_label = "other"
        else:
            # Convert prediction index to label name
            label_dict = load_label_mapping('label_dict.json')
            predicted_label = label_dict[preds.item()]

        # return {"predicted_label": predicted_label, "confidence": max_scores.item(), "prediction_scores": prediction.tolist()} ## for testing
        return {"message": "Classified successfully", "predicted_label": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"message": "Check your JSON: Ensure that newline characters are escaped (use \\n) and the JSON is properly formatted.", "details": exc.errors()},
    )

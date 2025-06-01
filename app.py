import logging
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import uvicorn

from config import CLASSIFICATION_MODEL_PATH, LOG_LEVEL, LOG_FORMAT
from pii_masking import mask_pii_details, nlp as spacy_nlp_model
from classification_model import classify_email_category, load_classification_model

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

app = FastAPI(title="Email Classification and PII Masking API by nitinprajwal", version="1.0.0")

# --- API Router for v1 --- #
router_v1 = APIRouter(prefix="/api/v1")

# --- Pydantic Models for API --- #
class EmailInput(BaseModel):
    input_email_body: str

class MaskedEntity(BaseModel):
    position: list[int]
    classification: str
    entity: str

class ClassificationOutput(BaseModel):
    input_email_body: str
    list_of_masked_entities: list[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# --- Load models --- #
logger.info("Loading PII NER model (spaCy) from pii_masking.py...")
pii_ner_model = spacy_nlp_model # Loaded from pii_masking.py
if pii_ner_model:
    logger.info("PII NER model (spaCy) loaded successfully.")
else:
    logger.error("PII NER model (spaCy) failed to load. PII masking for NER entities will not be available.")

logger.info(f"Loading classification model from {CLASSIFICATION_MODEL_PATH}...")
classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH) # Uses path from config
if classification_model is None:
    logger.critical("Email classification model could not be loaded. Classification will not be available.")
else:
    logger.info("Classification model loaded successfully.")


# --- Health Check Endpoint ---
@router_v1.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint called.")
    services = {
        "pii_ner_model_status": "loaded" if pii_ner_model else "not_loaded",
        "classification_model_status": "loaded" if classification_model else "not_loaded"
    }
    if pii_ner_model and classification_model:
        logger.info("Health check: All services OK.")
        return {"status": "ok", "services": services}
    else:
        service_issues = []
        if not pii_ner_model:
            service_issues.append("PII NER model not loaded")
        if not classification_model:
            service_issues.append("Classification model not loaded")
        
        logger.warning(f"Health check: Issues detected - {', '.join(service_issues)}")
        # Return 503 if critical services are down
        raise HTTPException(
            status_code=503, 
            detail={"status": "error", "message": "One or more critical services are unavailable.", "services": services}
        )

@router_v1.post("/classify", response_model=ClassificationOutput, tags=["Classification"])
async def classify_email_endpoint(email_input: EmailInput):
    logger.info(f"Received request for /classify. Email length: {len(email_input.input_email_body)}")
    if len(email_input.input_email_body) == 0:
        logger.warning("Received empty email body for /classify.")
        raise HTTPException(status_code=400, detail="Input email body cannot be empty.")
    original_email = email_input.input_email_body

    # 1. PII Masking
    if pii_ner_model is None:
        logger.warning("PII NER model (spaCy) not available at request time. Masking will be limited to regex-based detections.")
    
    logger.debug("Performing PII masking...")
    masked_email_text, pii_entities_raw = mask_pii_details(original_email, nlp_model=pii_ner_model)
    logger.debug(f"PII masking complete. Found {len(pii_entities_raw)} raw entities before output conversion.")
    
    # Convert pii_entities_raw (list of dicts) to list of MaskedEntity objects
    pii_entities_output = [
        MaskedEntity(
            position=entity['position'],
            classification=entity['classification'],
            entity=entity['entity']
        ) for entity in pii_entities_raw
    ]

    # 2. Classification
    if classification_model is None:
        logger.error("Classification model not available at request time. Returning error category.")
        category = "Error: Classifier not available"
        # If classification is critical, an HTTPException could be raised here.
    else:
        category = classify_email_category(masked_email_text, classification_model)


    # Original PII entities are available for potential future demasking.

    logger.info(f"Email classified as '{category}'. Total masked entities: {len(pii_entities_output)}. Returning response.")
    return ClassificationOutput(
        input_email_body=original_email,
        list_of_masked_entities=pii_entities_output, # Use the converted list of Pydantic models
        masked_email=masked_email_text,
        category_of_the_email=category
    )

# Include the router in the main app instance
app.include_router(router_v1)

if __name__ == "__main__":
    # Note: Hugging Face Spaces will use its own command to run the app.
    # This is for local testing.
    logger.info("Starting Uvicorn server for local development on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

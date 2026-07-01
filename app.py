import os
import io
import uuid
from typing import Dict, Any, Optional

import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from fastapi import FastAPI, File, UploadFile, Request, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates

# ----------------------------
# CONFIG
# ----------------------------
MODEL_WEIGHTS_PATH = "best_resnet18_fake_real.pth"
IMG_SIZE = 224

NEG_LABEL = "Fake"
POS_LABEL = "Real"
THRESHOLD = 0.5

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------------------
# FASTAPI + TEMPLATES
# ----------------------------
app = FastAPI(title="Fake Medicine Detection")
templates = Jinja2Templates(directory="templates")

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# MODEL (UNCHANGED)
# ----------------------------
def build_model_1logit() -> nn.Module:
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m

model = build_model_1logit().to(device)
model.eval()

if not os.path.exists(MODEL_WEIGHTS_PATH):
    print(f"⚠️ WARNING: Model weights not found: {MODEL_WEIGHTS_PATH}")
    print("   Put best_resnet18_fake_real.pth next to app.py")
else:
    state = torch.load(MODEL_WEIGHTS_PATH, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    print("✅ Model weights loaded (1-logit).")

# ----------------------------
# PREPROCESS (UNCHANGED)
# ----------------------------
preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ----------------------------
# PREDICT (UNCHANGED)
# ----------------------------
@torch.no_grad()
def predict_pil(img: Image.Image) -> Dict[str, Any]:
    img = img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    logit = model(x).squeeze()
    prob_pos = float(torch.sigmoid(logit).item())

    if prob_pos >= THRESHOLD:
        pred = POS_LABEL
        confidence = prob_pos
    else:
        pred = NEG_LABEL
        confidence = 1.0 - prob_pos

    return {
        "predicted_class": pred,
        "confidence": round(confidence, 6),
        "probs": {
            POS_LABEL: round(prob_pos, 6),
            NEG_LABEL: round(1.0 - prob_pos, 6),
        },
    }

# ----------------------------
# OPENFDA HELPER
# ----------------------------
def first_val(x: Any) -> Optional[str]:
    if isinstance(x, list) and x:
        return str(x[0])
    if isinstance(x, str):
        return x
    return None

def build_med_response(doc: Dict[str, Any], q: str) -> Dict[str, Any]:
    openfda = doc.get("openfda", {})
    return {
        "query": q,
        "brand_name": first_val(openfda.get("brand_name")),
        "generic_name": first_val(openfda.get("generic_name")),
        "manufacturer_name": first_val(openfda.get("manufacturer_name")),
        "product_type": first_val(openfda.get("product_type")),
        "route": first_val(openfda.get("route")),
        "purpose": first_val(doc.get("purpose")),
        "indications_and_usage": first_val(doc.get("indications_and_usage")),
        "warnings": first_val(doc.get("warnings")),
        "dosage_and_administration": first_val(doc.get("dosage_and_administration")),
        "do_not_use": first_val(doc.get("do_not_use")),
        "stop_use": first_val(doc.get("stop_use")),
        "storage_and_handling": first_val(doc.get("storage_and_handling")),
        "active_ingredient": first_val(doc.get("active_ingredient")),
        "inactive_ingredient": first_val(doc.get("inactive_ingredient")),
    }

def openfda_search(drug_name: str) -> Dict[str, Any]:
    q = drug_name.strip()
    if not q:
        return {"error": "Empty medicine name."}

    url = "https://api.fda.gov/drug/label.json"

    searches = [
        f'openfda.brand_name:"{q}"',
        f'openfda.generic_name:"{q}"',
        f'(openfda.brand_name:"{q}" OR openfda.generic_name:"{q}")',
        q
    ]

    for s in searches:
        params = {"search": s, "limit": 1}
        try:
            r = requests.get(url, params=params, timeout=12)
        except Exception as e:
            return {"error": f"OpenFDA request failed: {str(e)}"}

        if r.status_code == 200:
            js = r.json()
            results = js.get("results", [])
            if results:
                return build_med_response(results[0], q)

    return {"error": "No results found on OpenFDA for this medicine name."}

# ----------------------------
# DEMO HEALTH GUIDE DATA
# ----------------------------
health_data = {
    "fever": {
        "title": "Fever",
        "about": "Fever is a temporary rise in body temperature, often due to infection.",
        "medicines": [
            "Paracetamol (only as per label/doctor advice)",
            "ORS / plenty of fluids",
            "Rest and temperature monitoring"
        ],
        "food": [
            "Coconut water",
            "Khichdi / light meals",
            "Soup",
            "Fruits with water content"
        ],
        "exercise": [
            "Complete rest",
            "Light walking only after recovery"
        ],
        "doctor": "See a doctor if fever is very high, lasts more than 2–3 days, or comes with breathing trouble."
    },
    "cold": {
        "title": "Common Cold",
        "about": "Common cold is a viral infection affecting nose and throat.",
        "medicines": [
            "Steam inhalation",
            "Saline nasal drops",
            "Paracetamol if fever/body pain is present"
        ],
        "food": [
            "Warm soup",
            "Honey with warm water",
            "Citrus fruits",
            "Herbal tea"
        ],
        "exercise": [
            "Rest",
            "Breathing exercises",
            "Very light stretching"
        ],
        "doctor": "Consult a doctor if symptoms worsen, chest pain occurs, or cold lasts too long."
    },
    "cough": {
        "title": "Cough",
        "about": "Cough helps clear the throat and airways but may happen due to infection or irritation.",
        "medicines": [
            "Warm water",
            "Honey (for adults/older children where appropriate)",
            "Doctor-approved cough syrup"
        ],
        "food": [
            "Warm liquids",
            "Turmeric milk",
            "Soup",
            "Soft foods"
        ],
        "exercise": [
            "Rest",
            "Deep breathing exercises"
        ],
        "doctor": "See a doctor if cough lasts more than 1–2 weeks or comes with blood, wheezing, or fever."
    },
    "headache": {
        "title": "Headache",
        "about": "Headache can happen due to stress, dehydration, lack of sleep, or illness.",
        "medicines": [
            "Paracetamol as per label",
            "Hydration",
            "Rest in a quiet room"
        ],
        "food": [
            "Water",
            "Banana",
            "Nuts",
            "Light home-cooked meals"
        ],
        "exercise": [
            "Neck stretching",
            "Meditation",
            "Short walk"
        ],
        "doctor": "Consult a doctor if headache is severe, frequent, or with vomiting/vision problems."
    },
    "acidity": {
        "title": "Acidity",
        "about": "Acidity causes burning sensation in chest or stomach due to excess acid reflux.",
        "medicines": [
            "Antacid only as appropriate",
            "Avoid oily/spicy food",
            "Small frequent meals"
        ],
        "food": [
            "Banana",
            "Curd",
            "Oats",
            "Plain rice"
        ],
        "exercise": [
            "Walking after meals",
            "Avoid heavy exercise immediately after eating"
        ],
        "doctor": "See a doctor if acidity is frequent or there is chest pain, vomiting, or weight loss."
    },
    "diabetes": {
        "title": "Diabetes",
        "about": "Diabetes affects blood sugar control and needs regular monitoring.",
        "medicines": [
            "Use only doctor-prescribed medicine",
            "Regular sugar monitoring",
            "Never self-medicate"
        ],
        "food": [
            "High-fiber foods",
            "Salads",
            "Whole grains",
            "Low-sugar fruits"
        ],
        "exercise": [
            "Daily walking",
            "Light yoga",
            "Doctor-approved exercise routine"
        ],
        "doctor": "Regular doctor follow-up is important for diabetes management."
    }
}

def get_health_info(q: str) -> Dict[str, Any]:
    q = q.strip().lower()
    if not q:
        return {"error": "Please enter a disease name."}

    if q in health_data:
        return health_data[q]

    for k, v in health_data.items():
        if q in k or k in q:
            return v

    return {
        "error": "No demo health guide found for this disease. Try: fever, cold, cough, headache, acidity, diabetes."
    }

# ----------------------------
# ROUTES: UI
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_web(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Please upload a valid image file."},
            status_code=400,
        )

    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        ext = ".png"

    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    data = await file.read()
    with open(save_path, "wb") as f:
        f.write(data)

    try:
        img = Image.open(save_path)
        pred = predict_pil(img)
    except Exception:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": "Could not process the uploaded image."},
            status_code=400,
        )

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "prediction": pred["predicted_class"],
            "confidence": pred["confidence"],
            "probs": pred["probs"],
            "image_path": f"/uploads/{filename}",
        },
    )

@app.get("/uploads/{filename}")
def get_upload(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path)

@app.get("/medicine", response_class=HTMLResponse)
def medicine_page(request: Request, name: str = Query(...)):
    med = openfda_search(name)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "med": med,
            "searched_name": name
        },
    )

@app.get("/health-guide", response_class=HTMLResponse)
def health_guide_page(request: Request, disease: str = Query(...)):
    guide = get_health_info(disease)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "guide": guide,
            "searched_disease": disease
        },
    )

@app.post("/book-appointment", response_class=HTMLResponse)
def book_appointment(
    request: Request,
    patient_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    doctor_type: str = Form(...),
    date: str = Form(...),
    message: str = Form("")
):
    booking_msg = f"Appointment request submitted for {patient_name} with {doctor_type} on {date}."
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "booking_msg": booking_msg,
            "booking_data": {
                "patient_name": patient_name,
                "email": email,
                "phone": phone,
                "doctor_type": doctor_type,
                "date": date,
                "message": message,
            }
        },
    )

# ----------------------------
# ROUTES: JSON APIs
# ----------------------------
@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Could not read image."}, status_code=400)

    return predict_pil(img)

@app.get("/api/medicine-info")
def api_medicine_info(name: str = Query(...)):
    return openfda_search(name)

@app.get("/api/health-guide")
def api_health_guide(disease: str = Query(...)):
    return get_health_info(disease)
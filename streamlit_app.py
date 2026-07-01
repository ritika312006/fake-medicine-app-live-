import os
import io
from typing import Dict, Any, Optional

import requests
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

import streamlit as st

# ----------------------------
# CONFIG
# ----------------------------
MODEL_WEIGHTS_PATH = "best_resnet18_fake_real.pth"
IMG_SIZE = 224

NEG_LABEL = "Fake"
POS_LABEL = "Real"
THRESHOLD = 0.5

st.set_page_config(
    page_title="Fake Medicine Detection",
    page_icon="💊",
    layout="centered",
)

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# MODEL (cached so it only loads once per session)
# ----------------------------
@st.cache_resource(show_spinner="Loading detection model...")
def load_model():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    m = m.to(device)
    m.eval()

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        st.warning(
            f"⚠️ Model weights not found: {MODEL_WEIGHTS_PATH}. "
            "Predictions will not work until it is added to the app folder."
        )
        return m

    state = torch.load(MODEL_WEIGHTS_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    m.load_state_dict(cleaned, strict=True)
    return m


model = load_model()

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
        q,
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
            "Rest and temperature monitoring",
        ],
        "food": ["Coconut water", "Khichdi / light meals", "Soup", "Fruits with water content"],
        "exercise": ["Complete rest", "Light walking only after recovery"],
        "doctor": "See a doctor if fever is very high, lasts more than 2–3 days, or comes with breathing trouble.",
    },
    "cold": {
        "title": "Common Cold",
        "about": "Common cold is a viral infection affecting nose and throat.",
        "medicines": ["Steam inhalation", "Saline nasal drops", "Paracetamol if fever/body pain is present"],
        "food": ["Warm soup", "Honey with warm water", "Citrus fruits", "Herbal tea"],
        "exercise": ["Rest", "Breathing exercises", "Very light stretching"],
        "doctor": "Consult a doctor if symptoms worsen, chest pain occurs, or cold lasts too long.",
    },
    "cough": {
        "title": "Cough",
        "about": "Cough helps clear the throat and airways but may happen due to infection or irritation.",
        "medicines": ["Warm water", "Honey (for adults/older children where appropriate)", "Doctor-approved cough syrup"],
        "food": ["Warm liquids", "Turmeric milk", "Soup", "Soft foods"],
        "exercise": ["Rest", "Deep breathing exercises"],
        "doctor": "See a doctor if cough lasts more than 1–2 weeks or comes with blood, wheezing, or fever.",
    },
    "headache": {
        "title": "Headache",
        "about": "Headache can happen due to stress, dehydration, lack of sleep, or illness.",
        "medicines": ["Paracetamol as per label", "Hydration", "Rest in a quiet room"],
        "food": ["Water", "Banana", "Nuts", "Light home-cooked meals"],
        "exercise": ["Neck stretching", "Meditation", "Short walk"],
        "doctor": "Consult a doctor if headache is severe, frequent, or with vomiting/vision problems.",
    },
    "acidity": {
        "title": "Acidity",
        "about": "Acidity causes burning sensation in chest or stomach due to excess acid reflux.",
        "medicines": ["Antacid only as appropriate", "Avoid oily/spicy food", "Small frequent meals"],
        "food": ["Banana", "Curd", "Oats", "Plain rice"],
        "exercise": ["Walking after meals", "Avoid heavy exercise immediately after eating"],
        "doctor": "See a doctor if acidity is frequent or there is chest pain, vomiting, or weight loss.",
    },
    "diabetes": {
        "title": "Diabetes",
        "about": "Diabetes affects blood sugar control and needs regular monitoring.",
        "medicines": ["Use only doctor-prescribed medicine", "Regular sugar monitoring", "Never self-medicate"],
        "food": ["High-fiber foods", "Salads", "Whole grains", "Low-sugar fruits"],
        "exercise": ["Daily walking", "Light yoga", "Doctor-approved exercise routine"],
        "doctor": "Regular doctor follow-up is important for diabetes management.",
    },
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
# UI
# ----------------------------
st.title("💊 Fake Medicine Detection")
st.caption("AI-powered detection of counterfeit medicines, plus medicine info and a basic health guide.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Detect Medicine", "📋 Medicine Info", "🩺 Health Guide", "📅 Book Appointment"]
)

# ---- TAB 1: Detection ----
with tab1:
    st.subheader("Upload a medicine image")
    uploaded_file = st.file_uploader(
        "Choose an image (packaging, strip, label, etc.)",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image_bytes = uploaded_file.read()

        try:
            img = Image.open(io.BytesIO(image_bytes))
            with col1:
                st.image(img, caption="Uploaded image", use_container_width=True)

            with st.spinner("Analyzing image..."):
                result = predict_pil(img)

            with col2:
                pred = result["predicted_class"]
                conf = result["confidence"] * 100

                if pred == "Real":
                    st.success(f"✅ Prediction: **{pred}**")
                else:
                    st.error(f"⚠️ Prediction: **{pred}**")

                st.metric("Confidence", f"{conf:.2f}%")
                st.write("**Probabilities:**")
                st.progress(result["probs"]["Real"], text=f"Real: {result['probs']['Real']*100:.2f}%")
                st.progress(result["probs"]["Fake"], text=f"Fake: {result['probs']['Fake']*100:.2f}%")

            st.caption(
                "⚠️ This is a sample model for demonstration. Always verify with a pharmacist "
                "or official source before trusting any result."
            )
        except Exception:
            st.error("Could not process the uploaded image. Please try a different file.")

# ---- TAB 2: Medicine Info (OpenFDA) ----
with tab2:
    st.subheader("Look up a medicine")
    med_name = st.text_input("Enter medicine / brand / generic name", key="med_name")
    if st.button("Search Medicine", key="med_search_btn"):
        if med_name.strip():
            with st.spinner("Searching OpenFDA..."):
                med = openfda_search(med_name)
            if "error" in med:
                st.warning(med["error"])
            else:
                st.write(f"### {med.get('brand_name') or med.get('query')}")
                info_pairs = [
                    ("Generic Name", med.get("generic_name")),
                    ("Manufacturer", med.get("manufacturer_name")),
                    ("Product Type", med.get("product_type")),
                    ("Route", med.get("route")),
                    ("Purpose", med.get("purpose")),
                    ("Active Ingredient", med.get("active_ingredient")),
                    ("Inactive Ingredient", med.get("inactive_ingredient")),
                    ("Indications & Usage", med.get("indications_and_usage")),
                    ("Dosage & Administration", med.get("dosage_and_administration")),
                    ("Warnings", med.get("warnings")),
                    ("Do Not Use", med.get("do_not_use")),
                    ("Stop Use", med.get("stop_use")),
                    ("Storage & Handling", med.get("storage_and_handling")),
                ]
                for label, value in info_pairs:
                    if value:
                        st.markdown(f"**{label}:** {value}")
        else:
            st.warning("Please enter a medicine name.")

# ---- TAB 3: Health Guide ----
with tab3:
    st.subheader("Basic health guide (demo data)")
    disease = st.text_input(
        "Enter a condition (e.g. fever, cold, cough, headache, acidity, diabetes)",
        key="disease_name",
    )
    if st.button("Get Guide", key="guide_btn"):
        guide = get_health_info(disease)
        if "error" in guide:
            st.warning(guide["error"])
        else:
            st.write(f"### {guide['title']}")
            st.write(guide["about"])

            st.markdown("**Suggested care:**")
            for m in guide["medicines"]:
                st.markdown(f"- {m}")

            st.markdown("**Helpful foods:**")
            for f in guide["food"]:
                st.markdown(f"- {f}")

            st.markdown("**Exercise / activity:**")
            for e in guide["exercise"]:
                st.markdown(f"- {e}")

            st.info(guide["doctor"])

# ---- TAB 4: Appointment Booking (demo form) ----
with tab4:
    st.subheader("Book a doctor appointment (demo)")
    with st.form("appointment_form"):
        patient_name = st.text_input("Patient Name")
        email = st.text_input("Email")
        phone = st.text_input("Phone")
        doctor_type = st.selectbox(
            "Doctor Type", ["General Physician", "Pharmacist", "Pediatrician", "Cardiologist", "Other"]
        )
        date = st.date_input("Preferred Date")
        message = st.text_area("Message (optional)")
        submitted = st.form_submit_button("Book Appointment")

        if submitted:
            if patient_name and email and phone:
                st.success(
                    f"Appointment request submitted for **{patient_name}** with "
                    f"**{doctor_type}** on **{date}**. (This is a demo — no real booking is made.)"
                )
            else:
                st.warning("Please fill in name, email, and phone.")

st.divider()
st.caption(
    "⚠️ This project currently uses a sample model/dataset. For production-level accuracy, "
    "train on a verified real-vs-fake medicine image dataset before relying on results."
)

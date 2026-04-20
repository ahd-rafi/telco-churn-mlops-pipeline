"""
Serving layer – FastAPI REST API + Gradio web UI.

Endpoints
---------
GET  /          Health check (used by ALB)
POST /predict   JSON prediction endpoint
GET  /ui        Interactive Gradio form
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from src.serving.inference import predict

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict whether a telecom customer is likely to churn.",
    version="1.0.0",
)


@app.get("/")
def health():
    """Lightweight health probe for load-balancer checks."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    """Eighteen customer attributes expected by the model."""

    # Demographics
    gender: str
    Partner: str
    Dependents: str

    # Phone
    PhoneService: str
    MultipleLines: str

    # Internet
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str

    # Billing & contract
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

    # Numeric
    tenure: int
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def churn_prediction(data: CustomerData):
    """Return a churn prediction for a single customer record."""
    try:
        return {"prediction": predict(data.dict())}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Gradio web interface
# ---------------------------------------------------------------------------
def _gradio_predict(
    gender, Partner, Dependents, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges,
):
    """Bridge between Gradio form inputs and the inference pipeline."""
    payload = {
        "gender": gender, "Partner": Partner, "Dependents": Dependents,
        "PhoneService": PhoneService, "MultipleLines": MultipleLines,
        "InternetService": InternetService, "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport, "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies, "Contract": Contract,
        "PaperlessBilling": PaperlessBilling, "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    return str(predict(payload))


_demo = gr.Interface(
    fn=_gradio_predict,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),

        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"],
                    label="Multiple Lines", value="No"),

        gr.Dropdown(["DSL", "Fiber optic", "No"],
                    label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"],
                    label="Streaming Movies", value="Yes"),

        gr.Dropdown(["Month-to-month", "One year", "Two year"],
                    label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ], label="Payment Method", value="Electronic check"),

        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0,
                  minimum=0, maximum=200),
        gr.Number(label="Total Charges ($)", value=85.0,
                  minimum=0, maximum=10000),
    ],
    outputs=gr.Textbox(label="Prediction", lines=2),
    title="Telco Customer Churn Predictor",
    description=(
        "Enter customer details to predict churn risk. "
        "The model uses an XGBoost classifier trained on historical telecom data."
    ),
    examples=[
        # High-risk customer
        ["Female", "No", "No", "Yes", "No", "Fiber optic", "No", "No",
         "No", "No", "Yes", "Yes", "Month-to-month", "Yes",
         "Electronic check", 1, 85.0, 85.0],
        # Low-risk customer
        ["Male", "Yes", "Yes", "Yes", "Yes", "DSL", "Yes", "Yes",
         "Yes", "Yes", "No", "No", "Two year", "No",
         "Credit card (automatic)", 60, 45.0, 2700.0],
    ],
    theme=gr.themes.Base(),
)

# Mount Gradio inside the FastAPI app at /ui
app = gr.mount_gradio_app(app, _demo, path="/ui")

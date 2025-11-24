from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

app=FastAPI()

classifier = pipeline("text-classification", model="./symptom_model", tokenizer="./symptom_model")

class SymptomInput(BaseModel):
    symptoms: str



def check_red_flags(text):
    red_flags = ["chest pain","difficulty breathing", "fainting", "severe headache", "seizure", "bleeding heavily"]
    for flag in red_flags:
        if flag in text.lower():
            return "CRITICAL"
    return None                    

def generate_explanation(text, severity):  
    if severity == "CRITICAL":
        return "Your symptoms include red flag emergency indicators."
    elif severity == "MODERATE":
        return "Your symptoms may require medical consultation if they persist."
    else:
        return "Your symptoms appear mild and common."


def action_from_severity(sev):
    if sev=="CRITICAL":
        return "Seek emergency care immediately"
    elif sev=="MODERATE":
        return "Consult a doctor if symptoms last over 24-48 hours."
    else:
        return "Home care is usually sufficient."

@app.get("/favicon.ico")
def favicorn():
    return{"message":"No favicon available"}

@app.post("/predict-severity")
def predict_severity(data: SymptomInput):
    text= data.symptoms

    rule_output=check_red_flags(text)
    if rule_output:
        severity=rule_output
        confidence=1.0
    else:
        result=classifier(text)[0]
        sevrity=result["label"]
        confidence=float(result["score"])

    explanation = generate_explanation(text, severity)
    action = action_from_severity(severity)

    return{
        "severity": severity,
        "confidence": confidence,
        "explanation": explanation,
        "recommended_action": action
    }

@app.get("/")
def read_root():
    return {"message": "Welcome to the Symptom Severity Prediction API","docs":"/docs"}
    
import os, json, joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
SCALER_PATH  = os.path.join(BASE_DIR, 'models', 'scaler.pkl')
RESULTS_PATH = os.path.join(BASE_DIR, 'models', 'results.json')

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
with open(RESULTS_PATH) as f:
    results = json.load(f)

app = FastAPI(title="FraudGuard API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Transaction(BaseModel):
    amount: float = Field(..., example=142.50)
    hour:   int   = Field(..., example=14)
    V1:float=0;V2:float=0;V3:float=0;V4:float=0;V5:float=0;V6:float=0;V7:float=0
    V8:float=0;V9:float=0;V10:float=0;V11:float=0;V12:float=0;V13:float=0;V14:float=0
    V15:float=0;V16:float=0;V17:float=0;V18:float=0;V19:float=0;V20:float=0;V21:float=0
    V22:float=0;V23:float=0;V24:float=0;V25:float=0;V26:float=0;V27:float=0;V28:float=0

def build_features(t: Transaction):
    d=t.dict(); amt=d['amount']; hr=d['hour']
    vv=[d[f'V{i}'] for i in range(1,29)]
    row={f'V{i}':d[f'V{i}'] for i in range(1,29)}
    row.update({
        'Amount':amt,'hour':hr,
        'is_night':int(hr>=22 or hr<=5),'is_weekend':0,
        'log_amount':np.log1p(amt),'amount_squared':amt**2,
        'is_round_amt':int(amt%10==0),'is_small_amt':int(amt<5),'is_large_amt':int(amt>500),
        'v1_v2':d['V1']*d['V2'],'v3_v4':d['V3']*d['V4'],'v1_v3':d['V1']*d['V3'],
        'v7_v20':d['V7']*d['V20'],'v14_v17':d['V14']*d['V17'],
        'v_mean':np.mean(vv),'v_std':np.std(vv),'v_max':np.max(vv),'v_min':np.min(vv),
        'v_range':np.max(vv)-np.min(vv),
        'risk_score':(-d['V14']*.3-d['V17']*.2+d['V12']*.15+np.log1p(amt)*.1+int(hr>=22 or hr<=5)*.25),
    })
    df=pd.DataFrame([row])
    sc=['Amount','log_amount','amount_squared','risk_score','v_mean','v_std','v_max','v_min','v_range']
    sc=[c for c in sc if c in df.columns]
    df[sc]=scaler.transform(df[sc])
    return df

def risk_label(p):
    if p<.3: return "LOW"
    if p<.6: return "MEDIUM"
    if p<.85: return "HIGH"
    return "CRITICAL"

def explain(p, is_fraud, t):
    flags=[]
    if t.hour>=22 or t.hour<=5: flags.append("late-night transaction")
    if t.amount>500: flags.append("high amount")
    if t.amount<1:   flags.append("suspiciously small amount")
    if not flags: flags.append("normal pattern")
    return f"{'FRAUD' if is_fraud else 'Legitimate'}: {', '.join(flags)}. Probability: {p*100:.1f}%"

@app.get("/")
def root():
    return {"message":"FraudGuard API v2.0","best_model":results.get('best_model')}

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.get("/model/info")
def model_info():
    return {
        "best_model":        results.get('best_model'),
        "models":            results.get('models',{}),
        "shap_top_features": results.get('shap_top_features',{}),
    }

@app.post("/predict")
def predict(t: Transaction):
    try:
        df   = build_features(t)
        prob = float(model.predict_proba(df)[0][1])
        is_f = prob >= 0.5
        return {
            "is_fraud":          is_f,
            "fraud_probability": round(prob,4),
            "risk_level":        risk_label(prob),
            "confidence":        round(max(prob,1-prob),4),
            "model_used":        results.get('best_model','Unknown'),
            "explanation":       explain(prob,is_f,t),
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch")
def batch(transactions: list[Transaction]):
    out=[]
    for t in transactions:
        try:
            df=build_features(t); prob=float(model.predict_proba(df)[0][1])
            out.append({"is_fraud":prob>=.5,"fraud_probability":round(prob,4),"risk_level":risk_label(prob)})
        except Exception as e:
            out.append({"error":str(e)})
    return {"results":out,"total":len(out),"frauds_detected":sum(1 for r in out if r.get("is_fraud"))}

if __name__=='__main__':
    uvicorn.run("main:app",host="0.0.0.0",port=8001,reload=True)

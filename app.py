from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mangum import Mangum

# Charger le modèle pré-entraîné
model = joblib.load('model.pkl')

# Initialiser FastAPI
app = FastAPI()
handler = Mangum(app)

# Charger les données PCA et StandardScaler utilisés lors de l'entraînement du modèle
scaler = StandardScaler()
pca = PCA(n_components=0.95)

# Définir la structure de la requête
class TrafficInput(BaseModel):
    dur: float
    proto: str
    service: str
    state: str
    spkts: int
    dpkts: int
    sbytes: int
    dbytes: int
    rate: float
    sttl: int
    dttl: int
    sload: float
    dload: float
    sloss: int
    dloss: int
    sinpkt: float
    dinpkt: float
    sjit: float
    djit: float
    swin: int
    stcpb: int
    dtcpb: int
    dwin: int
    tcprtt: float
    synack: float
    ackdat: float
    smean: int
    dmean: int
    trans_depth: int
    response_body_len: int
    ct_srv_src: int
    ct_state_ttl: int
    ct_dst_ltm: int
    ct_src_dport_ltm: int
    ct_dst_sport_ltm: int
    ct_dst_src_ltm: int
    is_ftp_login: int
    ct_ftp_cmd: int
    ct_flw_http_mthd: int
    ct_src_ltm: int
    ct_srv_dst: int
    is_sm_ips_ports: int
    attack_cat: str  # Non utilisé pour la prédiction, mais présent dans le dataset
    label: int

# Définir l'endpoint de prédiction
@app.post("/predict")
def predict_traffic(input_data: TrafficInput):
    # Convertir les entrées en un tableau numpy
    data = np.array([[input_data.dur, input_data.proto, input_data.service, input_data.state,
                      input_data.spkts, input_data.dpkts, input_data.sbytes, input_data.dbytes,
                      input_data.rate, input_data.sttl, input_data.dttl, input_data.sload,
                      input_data.dload, input_data.sloss, input_data.dloss, input_data.sinpkt,
                      input_data.dinpkt, input_data.sjit, input_data.djit, input_data.swin,
                      input_data.stcpb, input_data.dtcpb, input_data.dwin, input_data.tcprtt,
                      input_data.synack, input_data.ackdat, input_data.smean, input_data.dmean,
                      input_data.trans_depth, input_data.response_body_len, input_data.ct_srv_src,
                      input_data.ct_state_ttl, input_data.ct_dst_ltm, input_data.ct_src_dport_ltm,
                      input_data.ct_dst_sport_ltm, input_data.ct_dst_src_ltm, input_data.is_ftp_login,
                      input_data.ct_ftp_cmd, input_data.ct_flw_http_mthd, input_data.ct_src_ltm,
                      input_data.ct_srv_dst, input_data.is_sm_ips_ports]])

    # Normaliser les données (en utilisant le même scaler que pour l'entraînement)
    data_normalized = scaler.transform(data)

    # Appliquer la réduction de dimensionnalité PCA (en utilisant le même PCA que pour l'entraînement)
    data_reduced = pca.transform(data_normalized)

    # Faire la prédiction
    prediction = model.predict(data_reduced)
    result = "Attack" if prediction[0] == 1 else "Normal"

    # Retourner la réponse JSON
    return JSONResponse({"prediction": result})

# Lancer l'application avec Uvicorn si ce fichier est exécuté directement
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

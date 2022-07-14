from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse

from static.utils import feature_distribution, train_model, get_logs, get_specific_logs, infer

VERSION: str = "0.0.1"

STATIC_PATH: str = "static"

origins = [
    "http://localhost:4007",
]

app = FastAPI()
app.mount("/static", StaticFiles(directory=STATIC_PATH), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Data(BaseModel):
    duration: int
    protocol_type: int
    service: int
    flag: int
    src_bytes: int
    dst_bytes: int
    land: int
    wrong_fragment: int
    urgent: int
    hot: int
    num_failed_logins: int
    logged_in: int
    num_compromised: int
    root_shell: int
    su_attempted: int
    num_root: int
    num_file_creations: int
    num_shells: int
    num_access_files: int
    num_outbound_cmds: int
    is_host_login: int
    is_guest_login: int
    count: int
    srv_count: int
    serror_rate: float
    srv_serror_rate: float
    rerror_rate: float
    srv_rerror_rate: float
    same_srv_rate: float
    diff_srv_rate: float
    srv_diff_host_rate: float
    dst_host_count: float
    dst_host_srv_count: float
    dst_host_same_srv_rate: float
    dst_host_diff_srv_rate: float
    dst_host_same_src_port_rate: float
    dst_host_srv_diff_host_rate: float
    dst_host_serror_rate: float
    dst_host_srv_serror_rate: float
    dst_host_rerror_rate: float
    dst_host_srv_rerror_rate: float
        

@app.get("/")
async def root():
    return JSONResponse({
        "statusText" : "Root Endpoint for Network Intrusion Detection API",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.get("/version")
async def version():
    return JSONResponse({
        "statusCode" : 200,
        "statusText" : "Network Intrusion Detection API Version Fetch Successful",
        "version" : VERSION,
    })


@app.get("/distribution/{feature_name}")
async def get_feature_distribution(feature_name: str):
    imageData = feature_distribution(feature_name)
    return JSONResponse({
        "statusText" : "Distribution Fetch Successfulr",
        "statusCode" : 200,
        "imageData" : imageData,
        "message" : f"Distribution of Feature '{feature_name}'"
    })


@app.get("/train")
async def train():
    auc_model_fold_name, acc_model_fold_name = train_model()
    return JSONResponse({
        "statusText" : "Training Complete",
        "statusCode" : 200,
        "best_auc_model": f"{auc_model_fold_name.split('_')[0]}, {auc_model_fold_name.split('_')[1]}",
        "best_acc_model": f"{acc_model_fold_name.split('_')[0]}, {acc_model_fold_name.split('_')[1]}",
    })


@app.get("/train/logs")
async def train_logs_specific_model():
    logs = get_logs()
    if logs is not None:
        return JSONResponse({
            "statusText" : "Log Fetch Complete",
            "statusCode" : 200,
            "logs" : logs,
        })
    else:
        return JSONResponse({
            "statusText" : "No LogFile Found",
            "statusCode" : 404,
        })


@app.get("/train/logs/{model_name}/{fold}")
async def train_logs_specific_model(model_name: str, fold: str):
    logs = get_specific_logs(model_name, int(fold))
    if logs is not None:
        return JSONResponse({
            "statusText" : "Log Fetch Complete",
            "statusCode" : 200,
            "logs" : logs,
        })
    else:
        return JSONResponse({
            "statusText" : "No Log Found",
            "statusCode" : 404,
        })
    

@app.get("/infer")
async def get_infer():
    return JSONResponse({
        "statusText" : "Inference Endpoint",
        "statusCode" : 200,
        "version" : VERSION,
    })


@app.post("/infer")
async def post_infer(data: Data):
    y_pred, y_pred_proba = infer([   
        data.duration,
        data.protocol_type,
        data.service,
        data.flag,
        data.src_bytes,
        data.dst_bytes,
        data.land,
        data.wrong_fragment,
        data.urgent,
        data.hot,
        data.num_failed_logins,
        data.logged_in,
        data.num_compromised,
        data.root_shell,
        data.su_attempted,
        data.num_root,
        data.num_file_creations,
        data.num_shells,
        data.num_access_files,
        data.num_outbound_cmds,
        data.is_host_login,
        data.is_guest_login,
        data.count,
        data.srv_count,
        data.serror_rate,
        data.srv_serror_rate,
        data.rerror_rate,
        data.srv_rerror_rate,
        data.same_srv_rate,
        data.diff_srv_rate,
        data.srv_diff_host_rate,
        data.dst_host_count,
        data.dst_host_srv_count,
        data.dst_host_same_srv_rate,
        data.dst_host_diff_srv_rate,
        data.dst_host_same_src_port_rate,
        data.dst_host_srv_diff_host_rate,
        data.dst_host_serror_rate,
        data.dst_host_srv_serror_rate,
        data.dst_host_rerror_rate,
        data.dst_host_srv_rerror_rate,
    ])

    if y_pred is not None and y_pred_proba is not None:
        return JSONResponse({
            "statusText": "Inference Complete", 
            "statusCode": 200, 
            "prediction": str(y_pred), 
            "probability": str(y_pred_proba[0, 1]),
        })
    else:
        return JSONResponse({
            "statusText" : "Error in performing inference",
            "statusCode" : 404,
        })
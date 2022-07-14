import os
import cv2
import base64
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from time import time
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_fscore_support

from .model import Model

SEED: int = 42
N_SPLITS:int = 5
STATIC_PATH: str = "static"
MODEL_SAVE_PATH: str = os.path.join(STATIC_PATH, "model_data")
if not os.path.exists(MODEL_SAVE_PATH): os.makedirs(MODEL_SAVE_PATH)


LE_1 = LabelEncoder()
LE_2 = LabelEncoder()
LE_3 = LabelEncoder()


def breaker(num: int=50, char: str="*") -> None:
    return ("\n" + num*char + "\n")


def encode_image_to_base64(header: str = "data:image/png;base64", image: np.ndarray = None) -> str:
    '''
        Encodes an image to base64 string.
    '''
    assert image is not None, "Image is None"
    _, imageData = cv2.imencode(".jpeg", image)
    imageData = base64.b64encode(imageData)
    imageData = str(imageData).replace("b'", "").replace("'", "")
    imageData = header + "," + imageData
    return imageData


def get_data() -> tuple:
    '''
        1. Reads the csv file stored in STATIC_PATH and returns a tuple of features and labels. 
        2. Performs oversampling of the 'not majority class' to balance the dataset.
    '''
    df = pd.read_csv(os.path.join(STATIC_PATH, "data/train.csv"))
    df["class"] = df["class"].map({"normal" : 0, "anomaly" : 1})

    df.protocol_type = LE_1.fit_transform(df.protocol_type)
    df.service = LE_2.fit_transform(df.service)
    df.flag = LE_3.fit_transform(df.flag)


    features = df.iloc[:, :-1].copy().values
    labels = df.iloc[:, -1].copy().values

    del df

    return features, labels


def feature_distribution(feature_name: str) -> str:
    '''
        Returns the distribution of the feature_name
    '''
    df = pd.read_csv(os.path.join(STATIC_PATH, "data/train.csv"))

    df["class"] = df["class"].map({"normal" : 0, "anomaly" : 1})

    df.protocol_type = LE_1.fit_transform(df.protocol_type)
    df.service = LE_2.fit_transform(df.service)
    df.flag = LE_3.fit_transform(df.flag)

    plt.figure(figsize=(12, 6))
    for c in set(df["class"]):
        sns.kdeplot(x=df[df["class"] == c][feature_name], label=str(c), shade=True)
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(STATIC_PATH, "temp.png"))
    plt.close()

    image = cv2.imread(os.path.join(STATIC_PATH, "temp.png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return encode_image_to_base64(image=image)


def get_scores(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    '''
        Calculates the accuracy, precision, recall, f1-score and roc-auc score and returns a tuple of the scores
    '''
    accuracy = accuracy_score(y_pred, y_true)
    try:
        auc = roc_auc_score(y_pred, y_true)
    except:
        auc = 0.0
    precision, recall, f_score, _ = precision_recall_fscore_support(y_pred, y_true)
    return accuracy, auc, precision, recall, f_score


def train_model() -> tuple:
    '''
        1. Trains a multitude of models and stores the model with best AUC score in a pickle file
        2. Saves a logs.txt file containing the training logs verbose
        3. Saves a logfile.pkl file containing the training logs which is used in /train-logs
    '''
    names: list = ["lgr", "knc", "gnb", "dtc", "etc", "abc", "gbc", "etcs", "rfc"]

    X, y = get_data()

    features = [i for i in range(X.shape[1])]

    feature_transformer = Pipeline(
        steps=[
            ("Simple_Imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("Standard_Scaler", StandardScaler())
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("features", feature_transformer, features),
        ]
    )

    best_auc: float = 0.0
    best_acc: float = 0.0

    logger: dict = {
        "name" : [],
        "fold" : [], 
        "accuracy" : [], 
        "auc" : [], 
        "precision_0" : [], 
        "precision_1" : [], 
        "recall_0" : [],
        "recall_1" : [], 
        "f_score_0" : [],
        "f_score_1" : [],
        "time_taken" : None,
    }

    file = open(os.path.join(MODEL_SAVE_PATH, "logs.txt"), "w+")

    start_time = time()
    for name in names:
        fold: int = 1
        folds: list = []
        accs: list = []
        aucs: list = []
        pre0s: list = []
        pre1s: list = []
        rec0s: list = []
        rec1s: list = []
        f1_0s: list = []
        f1_1s: list = []
        file.write(breaker())
        for tr_idx, va_idx in KFold(n_splits=N_SPLITS, random_state=SEED, shuffle=True).split(X):
            X_train, X_valid, y_train, y_valid = X[tr_idx], X[va_idx], y[tr_idx], y[va_idx]
            my_model = Model(name, preprocessor, SEED)
            my_model.model.fit(X_train, y_train)

            y_pred = my_model.model.predict(X_valid)
            acc, auc, pre, rec, f1 = get_scores(y_valid, y_pred)

            folds.append(fold)
            accs.append(acc)
            aucs.append(auc)
            pre0s.append(pre[0])
            pre1s.append(pre[1])
            rec0s.append(rec[0])
            rec1s.append(rec[1])
            f1_0s.append(f1[0])
            f1_1s.append(f1[1])

            file.write(f"{my_model.model_name}, {fold} --> Accuracy: {acc:.5f}, ROC-AUC: {auc:.5f}, Precision: {pre}, Recall: {rec}, F-Score: {f1}\n")
            if auc > best_auc:
                best_auc = auc
                auc_model_fold_name = f"{name}_{fold}"
                
                with open(os.path.join(MODEL_SAVE_PATH, f"best_auc_model.pkl"), "wb") as fp:
                    pickle.dump(my_model.model, fp)
                
            if acc > best_acc:
                best_acc = acc
                acc_model_fold_name = f"{name}_{fold}"
                
                with open(os.path.join(MODEL_SAVE_PATH, f"best_acc_model.pkl"), "wb") as fp:
                    pickle.dump(my_model.model, fp)
                
            fold += 1

        logger["name"].append(name)
        logger["fold"].append(folds)
        logger["accuracy"].append(accs)
        logger["auc"].append(aucs)
        logger["precision_0"].append(pre0s)
        logger["precision_1"].append(pre1s)
        logger["recall_0"].append(rec0s)
        logger["recall_1"].append(rec0s)
        logger["f_score_0"].append(f1_0s)
        logger["f_score_1"].append(f1_1s)
        
    time_taken = (time() - start_time)

    file.write(breaker())
    file.write(f"Best AUC Model : {auc_model_fold_name.split('_')[0]}, Best Fold : {auc_model_fold_name.split('_')[1]}")
    file.write(f"\nBest ACC Model : {acc_model_fold_name.split('_')[0]}, Best Fold : {acc_model_fold_name.split('_')[1]}")
    file.write(breaker())
    file.write(f"Time Taken : {time_taken / 60:2f} minutes")
    file.write(breaker())
    file.close()

    logger["time_taken"] = time_taken

    with open(os.path.join(MODEL_SAVE_PATH, f"logfile.pkl"), "wb") as fp: pickle.dump(logger, fp)
    return auc_model_fold_name, acc_model_fold_name


def get_logs() -> dict:
    '''
        Returns the logs stored in the file" logfile.pkl"
    '''
    if "logfile.pkl" in os.listdir(MODEL_SAVE_PATH):
        with open(os.path.join(MODEL_SAVE_PATH, f"logfile.pkl"), "rb") as fp:
            data = pickle.load(fp)
        return data
    else:
        return None

def get_specific_logs(model_name: str, fold: int) -> dict:
    '''
        Returns the logs stored in the file "logfile.pkl" for a specific model and fold
    '''

    names: list = ["lgr", "knc", "gnb", "dtc", "etc", "abc", "gbc", "etcs", "rfc"]
    folds: list = [1, 2, 3, 4, 5]

    if model_name not in names or fold not in folds:
        return None

    if "logfile.pkl" in os.listdir(MODEL_SAVE_PATH):
        with open(os.path.join(MODEL_SAVE_PATH, f"logfile.pkl"), "rb") as fp:
            data = pickle.load(fp)

        index_1 = data["name"].index(model_name)
        index_2 = data["fold"][index_1].index(fold)

        return{
            "name" : model_name,
            "fold" : fold,
            "accuracy" : data["accuracy"][index_1][index_2],
            "auc" : data["auc"][index_1][index_2],
            "precision_0" : data["precision_0"][index_1][index_2],
            "precision_1" : data["precision_1"][index_1][index_2],
            "recall_0" : data["recall_0"][index_1][index_2],
            "recall_1" : data["recall_1"][index_1][index_2],
            "f_score_0" : data["f_score_0"][index_1][index_2],
            "f_score_1" : data["f_score_1"][index_1][index_2],
            "time_taken" : data["time_taken"],
        }

    else:
        return None


def infer(data: list) -> tuple:
    if "best_auc_model.pkl" in os.listdir(MODEL_SAVE_PATH):
        data = np.array(data).reshape(1, -1)
        with open(os.path.join(MODEL_SAVE_PATH, f"best_auc_model.pkl"), "rb") as fp:
            model = pickle.load(fp)
        y_pred = model.predict(data)
        y_pred_proba = model.predict_proba(data)

        return y_pred, y_pred_proba
    else:
        return None, None
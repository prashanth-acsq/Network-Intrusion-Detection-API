import io
import cv2
import sys
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


def breaker(num: int=50, char: str="*") -> None:
    print("\n" + num*char + "\n")


def decode_image(imageData) -> np.ndarray:
    header, imageData = imageData.split(",")[0], imageData.split(",")[1]
    image = np.array(Image.open(io.BytesIO(base64.b64decode(imageData))))
    if len(image.shape) == 4:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    return header, image


def show_image(image: np.ndarray, cmap: str="gnuplot2", title: str=None) -> None:
    plt.figure()
    plt.imshow(cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB), cmap=cmap)
    plt.axis("off")
    if title: plt.title(title)
    figmanager = plt.get_current_fig_manager()
    figmanager.window.state("zoomed")
    plt.show()


def main():
    args_1: str = "--base-url"
    args_2: str = "--mode"

    base_url: str = "http://127.0.0.1:4006"
    mode: str = "/"

    if args_1 in sys.argv: base_url = sys.argv[sys.argv.index(args_1)+1]
    if args_2 in sys.argv: mode = sys.argv[sys.argv.index(args_2)+1]
    
    breaker()
    if mode == "/":
        url: str = base_url + mode
        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
        else:
            print("Error: " + str(response.status_code))

    elif mode == "version":
        url: str = base_url + "/" + mode
        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Version: {response.json()['version']}")
        else:
            print("Error: " + str(response.status_code))
        
    elif mode == "distribution":
        feature_name: str = sys.argv[sys.argv.index(args_2) + 2]

        url: str = base_url + "/" + mode + "/" + f"{feature_name}"

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            _, image = decode_image(imageData=response.json()["imageData"])
            print(f"Status Text: {response.json()['statusText']}")
            show_image(image=image, title=response.json()["message"])
        else:
            print("Error: " + str(response.status_code))

    elif mode == "train":
        url: str = base_url + "/" + mode

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
            breaker()
            print(f"Best AUC Model : {response.json()['best_auc_model']}")
            print(f"Best ACC Model : {response.json()['best_acc_model']}")
        else:
            print("Error: " + str(response.status_code))
    
    elif mode == "logs":
        model_name: str = sys.argv[sys.argv.index(args_2) + 2]
        fold: str = sys.argv[sys.argv.index(args_2) + 3]

        url: str = base_url + "/" + "train/logs" + "/" + model_name + "/" + fold

        response = requests.request(method="GET", url=url)
        if response.status_code == 200:
            print(f"Status Text: {response.json()['statusText']}")
            breaker()
            logs: dict = response.json()["logs"]
            for key, value in logs.items():
                if key != "name" :
                    if key == "fold":
                        print(f"{key.title():>11}: {int(value)}")
                    else:
                        print(f"{key.title():>11}: {float(value):.5f}")
                else:
                    print(f"{key.title():>11}: {value}")
        else:
            print("Error: " + str(response.status_code))
    
    elif mode == "infer":
        url: str = base_url + "/" + mode

        # 0.52000
        payload: dict = {
            "duration" : 9001,
            "protocol_type" : 1, 
            "service" : 32, 
            "flag" : 7, 
            "src_bytes" : 328713959, 
            "dst_bytes" : 2329438, 
            "land" : 1, 
            "wrong_fragment" : 2, 
            "urgent" : 0, 
            "hot" : 74, 
            "num_failed_logins" : 1, 
            "logged_in" : 1, 
            "num_compromised" : 150, 
            "root_shell" : 1, 
            "su_attempted" : 0, 
            "num_root" : 120, 
            "num_file_creations" : 5, 
            "num_shells" : 000, 
            "num_access_files" : 7, 
            "num_outbound_cmds" : 000, 
            "is_host_login" : 000, 
            "is_guest_login" : 000, 
            "count" : 430, 
            "srv_count" : 430, 
            "serror_rate" : 0.75, 
            "srv_serror_rate" : 0.15, 
            "rerror_rate" : 0.6, 
            "srv_rerror_rate" : 0.2, 
            "same_srv_rate" : 0.9, 
            "diff_srv_rate" : 0.1, 
            "srv_diff_host_rate" : 0.75, 
            "dst_host_count" : 0.8, 
            "dst_host_srv_count" : 0.9, 
            "dst_host_same_srv_rate" : 0.95, 
            "dst_host_diff_srv_rate" : 0.05, 
            "dst_host_same_src_port_rate" : 0.95, 
            "dst_host_srv_diff_host_rate" : 0.05, 
            "dst_host_serror_rate" : 0.9, 
            "dst_host_srv_serror_rate" : 0.1, 
            "dst_host_rerror_rate" : 0.8, 
            "dst_host_srv_rerror_rate" : 0.2, 
        }

        response = requests.request(method="POST", url=url, json=payload)
        if response.status_code == 200:
            print(f"Probability: {float(response.json()['probability']):.5f}")
        else:
            print("Error: " + str(response.status_code))
    
    breaker()


if __name__ == "__main__":
    sys.exit(main() or 0)
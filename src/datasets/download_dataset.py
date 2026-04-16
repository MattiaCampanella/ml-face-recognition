import os
import json
import zipfile
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

# Leggi le credenziali dal file credentials.json
credentials_path = os.path.join(os.path.dirname(__file__), "..", "..", "credentials.json")
credentials_path = os.path.abspath(credentials_path)

if not os.path.exists(credentials_path):
    raise FileNotFoundError(
        f"File delle credenziali non trovato: {credentials_path}\n"
        "Crea un file 'credentials.json' nella root del progetto con il formato:\n"
        '{"username": "<kaggle_user>", "api_key": "<kaggle_api_key>"}'
    )

with open(credentials_path, "r") as f:
    creds = json.load(f)

USER = creds["username"]
API_KEY = creds["api_key"]

owner = "debarghamitraroy"
dataset = "casia-webface"

url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{dataset}"

os.makedirs("data", exist_ok=True)
zip_path = os.path.join("data", "dataset.zip")

response = requests.get(url, auth=HTTPBasicAuth(USER, API_KEY), stream=True)

if response.status_code == 200:
    size = response.headers.get("content-length")
    total = int(size) if size else None 
    with open(zip_path, "wb") as f, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=zip_path,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    print("Download completato!")

    print("Estrazione in corso...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data")
    print("Estrazione completata!")

    os.remove(zip_path)
    print("File zip rimosso.")
else:
    print(f"Errore: {response.status_code}")
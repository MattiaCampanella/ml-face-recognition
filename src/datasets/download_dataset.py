import os
import zipfile
import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm

USER = "c0mplx"
API_KEY = "KGAT_eac472f3e765b7b0ac041f3846d4ee3b"

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
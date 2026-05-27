# Deploy — Face Clustering Web App

Guida completa per testare in locale e deployare su **Render** (backend) + **Vercel** (frontend).

---

## Indice

1. [Prerequisiti](#prerequisiti)
2. [Step 1 — Export modello ONNX](#step-1--export-modello-onnx)
3. [Step 2 — Test locale backend](#step-2--test-locale-backend)
4. [Step 3 — Test locale frontend](#step-3--test-locale-frontend)
5. [Step 4 — Test integrato locale](#step-4--test-integrato-locale)
6. [Step 5 — Upload modello su HuggingFace](#step-5--upload-modello-su-huggingface)
7. [Step 6 — Deploy backend su Render](#step-6--deploy-backend-su-render)
8. [Step 7 — Deploy frontend su Vercel](#step-7--deploy-frontend-su-vercel)
9. [Step 8 — Keep-alive Render](#step-8--keep-alive-render)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisiti

| Tool                | Versione min. | Installazione                                                         |
| ------------------- | ------------- | --------------------------------------------------------------------- |
| Python              | 3.10+         | [python.org](https://www.python.org/)                                 |
| Node.js             | 18+           | [nodejs.org](https://nodejs.org/)                                     |
| Docker              | 20+           | [docker.com](https://www.docker.com/) (opzionale, per test container) |
| Git                 | 2.30+         | già installato                                                        |
| Account HuggingFace | —             | [huggingface.co](https://huggingface.co/)                             |
| Account Render      | —             | [render.com](https://render.com/)                                     |
| Account Vercel      | —             | [vercel.com](https://vercel.com/)                                     |

---

## Step 1 — Export modello ONNX

Converti il checkpoint PyTorch in formato ONNX (necessario per il backend che NON usa PyTorch).

### 1.1 Installa dipendenze per l'export

```bash
# Dalla root del progetto
pip install torch torchvision onnxruntime numpy
```

### 1.2 Esegui l'export

```bash
python scripts/export_onnx.py --output backend/model.onnx
```

Se non hai il checkpoint `demo/best.pt` in locale, lo script lo scarica automaticamente da HuggingFace (`C0MPLX/triplet/best.pt`).

### 1.3 Output atteso

```
Loading model...
Exporting to ONNX...
Exported ONNX model to backend/model.onnx
Verifying ONNX output...
Max absolute difference: X.XXe-07
PASS: ONNX output matches PyTorch output.

Done! Model size: ~44.7 MB
Upload to HuggingFace: huggingface-cli upload C0MPLX/triplet backend/model.onnx model.onnx
```

> **Nota:** il file `backend/model.onnx` è nel `.gitignore` — non va committato.

---

## Step 2 — Test locale backend

### 2.1 Crea virtual environment

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2.2 Installa dipendenze

```bash
pip install -r requirements.txt
```

### 2.3 Crea file `.env`

Crea `backend/.env` con:

```env
MODEL_REPO_ID=C0MPLX/triplet
MODEL_FILENAME=model.onnx
MODEL_CACHE_DIR=./.model_cache
MAX_IMAGES=50
MAX_BATCH_SIZE=8
IMAGE_SIZE=224
ALLOWED_ORIGINS=["http://localhost:5173"]
```

> **`MODEL_CACHE_DIR`**: in locale usa un path relativo (`./.model_cache`). Il modello verrà cercato qui. Se non esiste, viene scaricato da HuggingFace.

### 2.4 (Opzionale) Usa il modello già esportato

Se hai già fatto lo Step 1, copia il modello nella cache locale per evitare il download:

```bash
mkdir -p .model_cache
cp model.onnx .model_cache/model.onnx
```

### 2.5 Avvia il server

```bash
uvicorn app.main:app --reload --port 8000
```

### 2.6 Verifica

```bash
# Health check
curl http://localhost:8000/health
# Risposta attesa: {"status":"ok","model_loaded":true}

# Test clustering (con immagini di esempio)
curl -X POST http://localhost:8000/cluster \
  -F "files=@../demo/examples/correct.jpg" \
  -F "files=@../demo/examples/incorrect.jpg" \
  -F 'params={"algorithm":"dbscan","eps":0.25,"min_samples":2}'
```

### 2.7 (Alternativa) Test con Docker

```bash
cd backend
docker build -t face-clustering-api .
docker run -p 8000:8000 \
  -e MODEL_REPO_ID=C0MPLX/triplet \
  -e MODEL_FILENAME=model.onnx \
  -e MODEL_CACHE_DIR=/opt/render/project/.model_cache \
  -e 'ALLOWED_ORIGINS=["http://localhost:5173"]' \
  face-clustering-api
```

---

## Step 3 — Test locale frontend

### 3.1 Installa dipendenze

```bash
cd frontend
npm install
```

### 3.2 Crea file `.env.local`

Crea `frontend/.env.local` con:

```env
VITE_API_URL=http://localhost:8000
```

### 3.3 Avvia dev server

```bash
npm run dev
```

Il frontend sarà disponibile su `http://localhost:5173`.

---

## Step 4 — Test integrato locale

1. **Terminale 1** — Backend:

   ```bash
   cd backend
   .venv\Scripts\activate   # o source .venv/bin/activate
   uvicorn app.main:app --reload --port 8000
   ```

2. **Terminale 2** — Frontend:

   ```bash
   cd frontend
   npm run dev
   ```

3. Apri `http://localhost:5173` nel browser

4. **Test funzionale:**
   - Trascina 3-5 immagini di volti (puoi usare quelle in `demo/examples/`)
   - Scegli algoritmo e parametri nella sidebar
   - Clicca "Run Clustering"
   - Verifica che i cluster vengano mostrati correttamente
   - Testa il download ZIP

---

## Step 5 — Upload modello su HuggingFace

Il backend su Render scaricherà il modello ONNX da HuggingFace. Devi caricarlo una sola volta.

### 5.1 Installa CLI HuggingFace

```bash
pip install huggingface-hub
huggingface-cli login
```

### 5.2 Upload

```bash
huggingface-cli upload C0MPLX/triplet backend/model.onnx model.onnx
```

### 5.3 Verifica

Apri `https://huggingface.co/C0MPLX/triplet` e controlla che `model.onnx` sia presente nei file del repo.

> **Se usi un tuo repo HuggingFace**, cambia `MODEL_REPO_ID` di conseguenza sia qui che nelle env di Render.

---

## Step 6 — Deploy backend su Render

### 6.1 Crea Web Service

1. Vai su [render.com/dashboard](https://dashboard.render.com/)
2. **New** → **Web Service**
3. Collega il repo GitHub `MattiaCampanella/ml-face-recognition`
4. Configura:

| Campo              | Valore                                   |
| ------------------ | ---------------------------------------- |
| **Name**           | `face-clustering-api`                    |
| **Region**         | `Frankfurt (EU Central)` o la più vicina |
| **Branch**         | `main`                                   |
| **Root Directory** | `backend`                                |
| **Runtime**        | `Docker`                                 |
| **Instance Type**  | `Free`                                   |

### 6.2 Environment Variables

Aggiungi queste variabili nella sezione **Environment** di Render:

| Key               | Value                              | Note                                           |
| ----------------- | ---------------------------------- | ---------------------------------------------- |
| `MODEL_REPO_ID`   | `C0MPLX/triplet`                   | Repo HuggingFace con il modello                |
| `MODEL_FILENAME`  | `model.onnx`                       | Nome file nel repo HF                          |
| `MODEL_CACHE_DIR` | `/opt/render/project/.model_cache` | Path persistente su Render                     |
| `MAX_IMAGES`      | `50`                               | Max immagini per richiesta                     |
| `MAX_BATCH_SIZE`  | `4`                                | Batch ridotto per risparmiare RAM              |
| `IMAGE_SIZE`      | `224`                              | Non modificare                                 |
| `ALLOWED_ORIGINS` | `["https://TUO-APP.vercel.app"]`   | URL del frontend Vercel (aggiorna dopo Step 7) |

> **Importante:** `MAX_BATCH_SIZE=4` su Render (non 8) per stare entro i 512MB.  
> **`ALLOWED_ORIGINS`** va aggiornato dopo il deploy del frontend con l'URL effettivo Vercel.

### 6.3 Health Check

Nella sezione **Health Check** di Render:

| Campo | Valore    |
| ----- | --------- |
| Path  | `/health` |

### 6.4 Disk (opzionale ma consigliato)

Per evitare il re-download del modello ad ogni re-deploy:

1. Vai nella tab **Disks** del servizio
2. **Add Disk**:
   - **Mount Path**: `/opt/render/project/.model_cache`
   - **Size**: `1 GB` (il minimo)

> Senza disk, il modello viene ri-scaricato da HF ad ogni restart (~45MB, ~10 secondi).  
> Con disk, è istantaneo.

### 6.5 Deploy

Clicca **Create Web Service**. Il primo deploy:

1. Builda il Docker image (~2-3 min)
2. Scarica il modello da HuggingFace (~10 sec)
3. Carica il modello ONNX in memoria

**URL finale**: `https://face-clustering-api.onrender.com` (o simile)

### 6.6 Verifica deploy

```bash
curl https://face-clustering-api.onrender.com/health
# {"status":"ok","model_loaded":true}
```

---

## Step 7 — Deploy frontend su Vercel

### 7.1 Import progetto

1. Vai su [vercel.com/dashboard](https://vercel.com/dashboard)
2. **Add New** → **Project**
3. Importa il repo `MattiaCampanella/ml-face-recognition`
4. Configura:

| Campo                | Valore                    |
| -------------------- | ------------------------- |
| **Framework Preset** | `Vite`                    |
| **Root Directory**   | `frontend`                |
| **Build Command**    | `npm run build` (default) |
| **Output Directory** | `dist` (default)          |

### 7.2 Environment Variables

Nella sezione **Environment Variables**:

| Key            | Value                                      | Environments        |
| -------------- | ------------------------------------------ | ------------------- |
| `VITE_API_URL` | `https://face-clustering-api.onrender.com` | Production, Preview |

> Sostituisci con l'URL effettivo di Render dallo Step 6.5.

### 7.3 Deploy

Clicca **Deploy**. Build ~30 secondi.

**URL finale**: `https://tuo-progetto.vercel.app`

### 7.4 Aggiorna CORS su Render

Ora che hai l'URL Vercel, torna su Render e aggiorna la variabile:

```
ALLOWED_ORIGINS=["https://tuo-progetto.vercel.app"]
```

Se vuoi anche il preview di Vercel (per branch di develop):

```
ALLOWED_ORIGINS=["https://tuo-progetto.vercel.app","https://tuo-progetto-git-*.vercel.app"]
```

Render fa re-deploy automatico al cambio env vars.

---

## Step 8 — Keep-alive Render

Il free tier di Render mette in sleep il servizio dopo 15 minuti di inattività. Per tenerlo sveglio:

### Opzione A — UptimeRobot (consigliato)

1. Crea account su [uptimerobot.com](https://uptimerobot.com/)
2. **Add New Monitor**:
   - **Type**: HTTP(s)
   - **URL**: `https://face-clustering-api.onrender.com/health`
   - **Interval**: 5 minutes
3. Salva

### Opzione B — Cron-job.org

1. Vai su [cron-job.org](https://cron-job.org/)
2. Crea un job:
   - **URL**: `https://face-clustering-api.onrender.com/health`
   - **Schedule**: ogni 5 minuti

### Opzione C — GitHub Actions (se preferisci infra-as-code)

Crea `.github/workflows/keep-alive.yml`:

```yaml
name: Keep Render alive
on:
  schedule:
    - cron: "*/5 * * * *"
jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - run: curl -sf https://face-clustering-api.onrender.com/health
```

---

## Troubleshooting

### Backend non si avvia su Render

- Controlla i **Logs** nella dashboard Render
- Errore comune: `MODEL_CACHE_DIR` non scrivibile → assicurati che il Dockerfile crei la directory (`mkdir -p`)
- Se il modello non si scarica: verifica che `MODEL_REPO_ID` sia corretto e il repo HF sia pubblico

### CORS error nel browser

- Verifica che `ALLOWED_ORIGINS` su Render contenga esattamente l'URL del frontend (con `https://`, senza trailing slash)
- Formato corretto: `["https://tuo-progetto.vercel.app"]`

### Out of Memory su Render (512MB)

- Riduci `MAX_BATCH_SIZE` a `2`
- Riduci `MAX_IMAGES` a `30`
- Verifica nei Render Logs il picco di memoria

### Modello viene riscaricato ad ogni restart

- Aggiungi un **Disk** su Render (vedi Step 6.4)
- Senza disk, la cache si perde ad ogni deploy (ma il download è veloce ~10s)

### Frontend non si connette al backend

- Verifica che `VITE_API_URL` su Vercel sia corretto (senza trailing slash)
- Verifica che l'URL Render risponda: `curl https://tuo-url.onrender.com/health`
- Se Render è in sleep, la prima richiesta impiega ~30s per il cold start

### Le immagini non vengono clusterizzate correttamente

- Le immagini devono contenere **un singolo volto crop** (no foto di gruppo)
- Prova a regolare i parametri:
  - DBSCAN: abbassa `eps` per cluster più stretti, alza per più permissivi
  - Agglomerative: abbassa `threshold` per cluster più stretti

---

## Riepilogo variabili d'ambiente

### Backend (`backend/.env` locale / Render env vars)

```env
MODEL_REPO_ID=C0MPLX/triplet
MODEL_FILENAME=model.onnx
MODEL_CACHE_DIR=./.model_cache                    # locale
# MODEL_CACHE_DIR=/opt/render/project/.model_cache  # Render
MAX_IMAGES=50
MAX_BATCH_SIZE=8                                   # locale
# MAX_BATCH_SIZE=4                                  # Render (meno RAM)
IMAGE_SIZE=224
ALLOWED_ORIGINS=["http://localhost:5173"]           # locale
# ALLOWED_ORIGINS=["https://tuo-app.vercel.app"]    # produzione
```

### Frontend (`frontend/.env.local` locale / Vercel env vars)

```env
VITE_API_URL=http://localhost:8000                  # locale
# VITE_API_URL=https://face-clustering-api.onrender.com  # produzione
```

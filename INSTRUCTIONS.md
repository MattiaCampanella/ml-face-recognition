# Track 1: Metric Learning for Face Recognition
**Suggested Size**: Small  
**Reference Module**: Metric Learning  

#### Problem Description
Face recognition systems are ubiquitous, but identifying individuals accurately across varying lighting, poses, and expressions is challenging. This project involves learning robust, identity-preserving embeddings by training a model on a large face dataset and testing its generalization on unseen individuals. Split the dataset into a training set and a test set, where the test set contains only individuals that are not present in the training set.

#### Dataset
- **CASIA-WebFace** (https://www.kaggle.com/datasets/debarghamitraroy/casia-webface) or a subset.
- ~500,000 images representing ~10,000 subjects.

#### Minimum Objectives
1. **Baseline**: Train a standard image classification model using a fine-tuned ResNet-18 backbone, relying on the classifier's features and a K-Nearest Neighbors (KNN) search to generalize to new faces.
2. **Metric Learning**: Implement Triplet Loss with hard negative mining. Given an anchor face, dynamically mine challenging positives (same person) and negatives (different people).
3. **Retrieval Evaluation**: Compute mAP @1, 5, 10 to evaluate if the model successfully retrieves the correct identity within its top rank predictions.
4. **Cluster Analysis**: Perform an analysis of the latent space (e.g., using t-SNE or PCA) to verify that faces of identical subjects cluster closely together.

#### Extra Objectives
- Implement and compare with advanced margin-based loss functions (e.g., ArcFace, CosFace, Siamese networks).
- Perform ablations on hyper-parameters like mining strategies (offline vs. online) and batch size.
- Create a small demo that processes and recognizes localized facial images collected directly by your team.

---

### 📏 Extra Notes
Here are the steps to get started:
1. **Choose a project**: Consult the [project list below](#project-list) to read the available tracks and check which ones are free or already assigned. Then communicate your chosen project to the professor via email.
2. **Fork**: Create a **fork** of this repository in your personal GitHub account. While it is preferable to use the Fork button in the top right (to keep the history visible for evaluation), you can also create a standalone repository and keep it private if you prefer.
3. **Clone**: Clone your fork locally.
4. **Work in this root**: Consult [CONTRIBUTING.md](CONTRIBUTING.md) for the required conventions on how to structure folders (`src/`, `data/`, `notebooks/`), how to write clean code, and how to use Git professionally in a team. Replace the placeholders in the `README.md` file with the technical information about the repository and `docs/REPORT.md` with the description of the project and the work done.
5. **AI Usage Policy**: The use of generative AI tools (ChatGPT, GitHub Copilot, Claude, etc.) is **permitted, but regulated**. The use of these tools is encouraged to speed up boilerplate code writing, for debugging, or as documentation support. However, **never delegate strategic thinking and architectural choices to AI**. Elaborate your strategy, write or generate the code, and take full responsibility for every line. The use of such tools must be explicitly declared in the final report.
6. **License**: It is good practice to release your work open source. You will find a `LICENSE` file (pre-set to MIT license). Open the file, replace `[Year]` and `[Name and Surname]` with the current year and the members of your team. Remember to choose a different one if you do not want to freely share your code.
7. **Submission**: Your GitHub fork is the **final deliverable** of the project. Ensure the code is reproducible following the instructions below and that the slides for the exam presentation are placed inside the `docs/` folder. If you opted for a private repository, evaluation can take place by making the repo visible to the professor (handle `antoninofurnari`) or by sending the repository source code via email.

---

### Plan

Obiettivo: implementare una pipeline riproducibile end-to-end per riconoscimento facciale su identità non viste, includendo baseline classificativa con feature retrieval, metrica Triplet con hard negative mining, valutazione retrieval con mAP@1/5/10 e analisi dello spazio latente con PCA/t-SNE. Approccio: costruire prima un baseline stabile (dati, config, train, eval), poi estendere con metric learning e infine consolidare risultati/documentazione.

### Steps
1. Fase 1 - Fondazioni progetto (settimana 1): aggiornare ambiente e infrastruttura comune. Creare schema configurazioni YAML centralizzato, utility di seed/device/logging/checkpoint, convenzioni cartelle output in experiments. Questa fase sblocca tutte le successive.
2. Fase 1 - Fondazioni progetto (settimana 1): implementare parser dataset CASIA-WebFace con split per identità disgiunte train/val/test (nessuna identità condivisa). Salvare split versionati per riproducibilità. Dipende da 1.
3. Fase 2 - Baseline classificativa (settimane 2-3): implementare modello baseline ResNet-18 fine-tuned con head di classificazione, training loop supervisionato, export embedding da backbone e retrieval KNN/cosine su test unseen. Dipende da 1-2.
4. Fase 2 - Baseline classificativa (settimane 2-3): implementare metrica retrieval mAP@1/5/10 con validazione su casi toy deterministici e salvataggio risultati JSON/CSV. Dipende da 3; in parallelo con 5.
5. Fase 2 - Baseline classificativa (settimane 2-3): implementare modulo clustering (PCA e t-SNE) e script di plotting per verificare separazione identità nello spazio embedding baseline. Dipende da 3; parallel con 4.
6. Fase 3 - Metric learning (settimane 3-4): integrare embedding head dedicata e Triplet Loss con mining hard online batch-wise (hard positive e hard negative nel batch). Introdurre sampler PK (P identità x K immagini) per garantire mining efficace. Dipende da 2-3.
7. Fase 3 - Metric learning (settimane 3-4): addestrare varianti minime richieste (baseline vs triplet) con config separati, checkpoint best/final e logging comparabile. Dipende da 6.
8. Fase 4 - Confronto e analisi (settimana 5): rieseguire valutazione retrieval e clustering su modello triplet, confrontare con baseline in tabelle finali, raccogliere esempi qualitativi top-k corretti/errati. Dipende da 4-5-7.
9. Fase 4 - Reproducibilità e consegna (settimana 6): completare README con comandi reali train/eval, completare REPORT con metodologia/risultati/limiti, dichiarare uso AI, verifica run pulito da ambiente nuovo. Dipende da 8.

### Verification
1. Verifica dati: script check che conferma disgiunzione identità tra split e conta immagini/identità per split.
2. Verifica training baseline: avvio train con baseline config su subset ridotto, controllo che loss decresca e checkpoint venga scritto.
3. Verifica retrieval baseline: esecuzione evaluate baseline con output mAP@1/5/10 e file risultati persistiti.
4. Verifica clustering baseline: generazione plot PCA e t-SNE con etichette identità campionate.
5. Verifica triplet mining: test unitari su miner (distanze e selezione hard positiva/negativa corrette).
6. Verifica training triplet: train completo con PK sampling e controllo stabilità loss triplet.
7. Verifica confronto finale: tabella unica baseline vs triplet (mAP@1/5/10) e confronto qualitativo retrieval top-k.
8. Verifica riproducibilità: da ambiente pulito, run dei comandi README senza patch manuali.

### Decisions
- Incluso: esclusivamente Minimum Objectives richiesti dal track.
- Escluso: ArcFace/CosFace, ablation estese, demo applicativa realtime.
- Scelta tecnica consigliata: prima implementazione semplice e trasparente; eventuale dipendenza a librerie metric-learning solo se accelera senza opacizzare la metodologia.
- Vincolo chiave: split per identità unseen obbligatorio e tracciato su file versionato.

### Further Considerations
1. Dimensione subset iniziale consigliata: partire con 10-20% del dataset per debugging veloce, poi scalare al massimo gestibile su GPU.
2. Strategia retrieval consigliata: cosine similarity su embedding L2-normalizzati come default; KNN euclidea come baseline secondaria.
3. Criterio di model selection: best checkpoint selezionato su mAP@5 validation, non solo su training loss.

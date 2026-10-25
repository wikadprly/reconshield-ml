# ReconShield ML Service

Python side of the ReconShield research project — **Random Forest training,
preprocessing, and the inference/metrics HTTP service** consumed by the
Next.js dashboard (`reconshield-web`).

The two repositories are deliberately separate:

| Folder | Stack | Responsibility |
|---|---|---|
| `reconshield-ml/` | Python (FastAPI + scikit-learn) | Train the model, serve inference, expose metrics |
| `reconshield-web/` | Next.js 16 + PostgreSQL | Dashboard UI, service layer, MikroTik orchestration |

## Folder layout

```
reconshield-ml/
├── api/
│   └── server.py         # FastAPI app: GET /model/metrics, POST /predict
├── training/
│   ├── train_rf.py       # Train RandomForest -> models/*.joblib + metrics.json
│   └── sync_metrics_db.py# Push latest metrics.json into PostgreSQL
├── preprocess/
│   └── features.py       # Flow record -> numeric feature vector (shared logic)
├── data/
│   ├── raw/              # Original CSV / NetFlow exports (git-ignored)
│   └── processed/        # Cleaned splits + metrics.json (git-ignored)
├── models/               # *.joblib artifacts (git-ignored)
├── notebooks/            # CRISP-DM notebook (crisp_dm.ipynb) — dokumentasi laporan
├── config.py             # Single source of truth for paths + env config
├── requirements.txt
└── .env.example
```

## Setup

```bash
cd reconshield-ml
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env          # Windows (set DATABASE_URL, MODEL_PATH, ...)
```

## 1. Prepare training data

**Dataset utama (sudah ada):** `../dataset/ta-recon/data/dataset.csv` — 6.404 flow
nyata dari testbed (NetFlow v5), hasil pipeline upstream di `dataset/ta-recon/`:

```
netflow_v5_collector.py → data/flows.csv → build_dataset.py → data/dataset.csv
```

CSV harus memuat:

- **16 kolom fitur per-flow** — persis `preprocess/features.py::FEATURE_COLUMNS`
  (`prot`, `tos`, `tcp_flags`, `sport`, `dport`, `packets`, `bytes`, `duration_ms`,
  `bytes_per_pkt`, `pkts_per_sec`, `f_fin`, `f_syn`, `f_rst`, `f_psh`, `f_ack`, `f_urg`).
  Kolom IP (`src`, `dst`, `exporter`) boleh ada di CSV tetapi **tidak** pernah jadi fitur
  (anti kebocoran subnet — aturan yang sama dengan `build_dataset.py`).
- **Target column** — `label`; nilai `Normal`/`Reconnaissance` (alias `normal`, `recon`,
  `scan`, `attack`, ... juga diterima dan dinormalisasi otomatis).

```bash
python -m training.train_rf --data ../dataset/ta-recon/data/dataset.csv --target label --version v1.1.0
```

> ⚠️ `data/raw/smoke_flows.csv` adalah sampel format lama (10 fitur) sebelum kontrak
> disamakan ke 16 fitur dataset nyata — **jangan dipakai** untuk training; kalau dilatih
> dari file itu semua fitur akan jadi nol secara diam-diam.

Outputs:
- `models/Random_Forest_Classifier_<version>.joblib`
- `data/processed/metrics.json` — **exactly the `ModelMetrics` contract the web reads**

Artefak CRISP-DM untuk laporan: jalankan `notebooks/crisp_dm.ipynb` (menyajikan data
understanding → evaluasi dengan parameter & angka yang sama dengan `metrics.json`).

## 2. Serve the metrics/inference API

```bash
uvicorn api.server:app --reload --port 8000
```

The web app discovers this service via `PYTHON_ML_API_URL` (e.g. `http://localhost:8000`).
Its `ml-service.ts` currently calls `GET /model/metrics` — keep that path stable.

```bash
curl http://localhost:8000/model/metrics
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_flow.json
```

## 3. Automatic detection pipeline (NetFlow → model → database)

Loop otomatis proposal §6.2 langkah 3–6 & 11 (temuan A3): terima NetFlow v5,
 ekstrak 16 fitur yang sama, klasifikasi dengan artefak model yang sama dengan API,
 tulis insiden ke tabel `detections` (label `Reconnaissance` saja secara default).

```bash
python -m pipeline.detect                 # UDP 2055, butuh DATABASE_URL
python -m pipeline.detect --dry-run       # klasifikasi tanpa menulis DB
python -m pipeline.simulate_netflow       # uji lokal tanpa lab (kirim paket v5 karangan)
```

- `DATABASE_URL` diambil dari env atau `.env` — boleh persis salinan dari
  `reconshield-web/.env`; parameter khas Prisma (`?schema=...`) otomatis dibersihkan
  oleh `config._for_psycopg` sebelum dipakai psycopg2.
- Insiden ditulis dengan `action=Pending` / `status=Review` — **bukan** `TARPIT`:
  eksekusi mitigasi (A4) menunggu keputusan desain A20 (`docs/keputusan-a20-tarpit.md`).
- Anti banjir: de-duplikasi per IP sumber (`--dedup-seconds`, default 60) dan port noise
  (2055/514/8291) dilewati — konsisten dengan `build_dataset.py`.
- Log keluaran memuat latensi inference (rata-rata/p95/max) saat proses dihentikan.

Terverifikasi end-to-end 25 Okt 2026 (test case BB-M1…BB-M5 di
`../docs/black-box-testing.md`): simulator → pipeline → `INC-1057` masuk database.

## 4. Sync metrics into PostgreSQL (optional but recommended)

Once trained, publish the numbers so the dashboard reads them from the
`model_evaluations` table (source `database`) instead of hitting this API:

```bash
python -m training.sync_metrics_db
```

Requires `DATABASE_URL` in `.env`. Re-running with the same `version` will
upsert (update) rather than create duplicates.

## Integration with reconshield-web

In `reconshield-web/.env`:

```env
PYTHON_ML_API_URL="http://localhost:8000"
RECONSHIELD_USE_MOCK_DATA="true"   # set to "false" once the service is live
```

The web service layer (`security-service.ts`) resolves data in this order for
ML metrics: **PostgreSQL → this API → mock data**. Every result is labelled
with its `source` so the UI is honest about where numbers come from.

## Notes

- Do **not** commit `data/`, `models/`, or `.env` — they are git-ignored.
- Keep the `ModelMetrics` JSON shape in `api/server.py` in sync with
  `reconshield-web/src/types/security.ts`. If you add a metric field, update
  **both** files (the web only reads fields it knows about).
- The web app is currently in a **read-only phase**; inference results are
  displayed but the write path to MikroTik is intentionally disabled.

# Early Detection and Severity Classification of Diabetic Retinopathy Using Deep Learning

An AI-powered web application for screening diabetic retinopathy from retinal fundus images using EfficientNet-B0 transfer learning and Grad-CAM explainable AI.

> **Disclaimer:** This system is an educational/research prototype and is NOT intended to provide medical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## Features

- Deep learning-based 5-class diabetic retinopathy classification
- EfficientNet-B0 transfer learning with fine-tuning
- Grad-CAM explainable AI heatmaps
- Modern responsive web dashboard
- Prediction history with SQLite database
- Risk level assessment
- Interactive Chart.js visualizations
- Drag-and-drop image upload
- Demo mode for UI testing

## Technologies

| Layer | Technology |
|-------|-----------|
| Deep Learning | TensorFlow, Keras, EfficientNet-B0 |
| Computer Vision | OpenCV, Pillow |
| Backend | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Chart.js |
| Database | SQLite |
| Explainable AI | Grad-CAM |

## Project Structure

```
diabetic-retinopathy-ai/
├── app.py                  # Flask application
├── config.py               # Configuration
├── database.py             # SQLite database
├── requirements.txt
├── ml/
│   ├── prepare_data.py     # Dataset loading & validation
│   ├── train.py            # Model training
│   ├── evaluate.py         # Model evaluation
│   ├── predict.py          # Prediction
│   ├── gradcam.py          # Grad-CAM explainability
│   └── utils.py            # Utilities
├── data/                   # Dataset (APTOS 2019)
├── models/                 # Saved models
├── results/                # Evaluation outputs
├── database/               # SQLite database
├── uploads/                # Uploaded images
├── static/
│   ├── css/style.css
│   └── js/app.js
└── templates/
    ├── base.html
    ├── index.html
    ├── dashboard.html
    ├── analysis.html
    ├── result.html
    ├── history.html
    └── about.html
```

## Dataset Setup

Download the **APTOS 2019 Blindness Detection** dataset from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection).

Place files in the `data/` directory:
```
data/
├── train.csv
├── train_images/
│   ├── *.png
└── test_images/
```

The CSV should contain columns: `id`, `diagnosis`

## Installation

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Dataset

Place your APTOS 2019 dataset in the `data/` directory, or edit `config.py` to point to your dataset location.

## Training

```bash
python ml/train.py
```

This will:
1. Load and validate the dataset
2. Build EfficientNet-B0 model
3. Train the classification head (frozen backbone)
4. Fine-tune upper layers
5. Save the best model to `models/best_model.keras`

## Evaluation

```bash
python ml/evaluate.py
```

Generates:
- `results/metrics.json` - Full evaluation metrics
- `results/confusion_matrix.png` - Confusion matrix
- `results/accuracy_loss.png` - Training curves
- `results/per_class_performance.png` - Per-class metrics
- `results/classification_report.txt` - Detailed report

## Running the Application

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Deployment (Vercel + ML API)

The system is split into two parts because TensorFlow cannot run inside Vercel's
serverless functions:

- **`api.py`** — Flask API backend (prediction, history, statistics, media files).
  Deploy on **Render** (or any Python host that supports a persistent disk and
  can run TensorFlow). Includes `render.yaml` and `api-requirements.txt`.
- **`deploy/vercel/`** — static frontend (plain HTML/CSS/JS + Chart.js from CDN).
  Deploy the folder on **Vercel** as a static site. `vercel.json` proxies every
  `/api/*` request to the backend, so no CORS is needed in production.

### 1. Deploy the ML API

Option A — Render blueprint:

1. Push this repository to GitHub.
2. In Render, create a **Blueprint** from the repo (it reads `render.yaml`).
3. The service starts via `gunicorn api:app`. Set the `RA_DATA_ROOT` env var to
   the mounted disk path (`/opt/data` in the blueprint) so history/uploads persist.
4. Note the service URL, e.g. `https://retinaai-api.onrender.com`.

Option B — run locally: `python api.py` (serves on `http://127.0.0.1:5000`).

### 2. Deploy the frontend on Vercel

```bash
cd deploy/vercel
python build.py        # regenerate pages after editing ./src
npm i -g vercel
vercel                  # link project, upload, done
```

1. Edit `vercel.json` if your backend URL differs.
2. The site is fully static. `js/config.js` calls `/api/*` in production
   (rewritten by Vercel to the backend) or `http://127.0.0.1:5000/api` when
   opened from `localhost`.

### 3. Local split test

```bash
python api.py                          # terminal 1 — API on :5000
python -m http.server 8899 --directory deploy/vercel   # terminal 2 — static site
```

Open http://127.0.0.1:8899 in your browser.

## Demo Mode

For UI testing without a trained model, set `DEMO_MODE = True` in `config.py`. This simulates predictions and is clearly labeled in the interface.

## Key Pages

| Page | Description |
|------|-------------|
| Dashboard | Overview with stats and charts |
| Analysis | Image upload and prediction |
| History | All previous analyses |
| About | System information and disclaimers |

## Model Architecture

```
Input (224×224×3)
→ EfficientNet-B0 (ImageNet weights)
→ Global Average Pooling
→ Dropout (0.3)
→ Dense (256, ReLU)
→ Dropout (0.3)
→ Dense (5, Softmax)
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Batch Size | 32 |
| Initial LR | 1e-3 |
| Fine-tune LR | 1e-5 |
| Dropout | 0.3 |
| Head Epochs | 30 |
| Fine-tune Epochs | 15 |

All parameters are configurable in `config.py`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python ml/train.py` first |
| Dataset not found | Place APTOS dataset in `data/` directory |
| CUDA errors | Install `tensorflow-gpu` or use CPU mode |
| Port in use | Change `FLASK_PORT` in `config.py` |
| Import errors | Activate virtual environment |

## Model Limitations

- Trained on APTOS 2019 dataset (Indian population)
- Performance may vary with different camera types and image qualities
- Not validated for clinical use
- May have bias toward training data demographics

## Healthcare Disclaimer

This system is a college/research prototype for educational purposes. It should NOT be used as a substitute for professional medical diagnosis. AI-generated results are for educational/research purposes only. Always consult a qualified ophthalmologist for professional evaluation.

## License

Educational use only.

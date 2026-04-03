# Weather Image Classification — MLOps Summative

## 🎥 Video Demo
[YouTube Demo Link — add after recording]

## 🌐 Live URL
https://summative-assignment-mlop-sm12.onrender.com


## 📋 Project Description
End-to-end MLOps pipeline for **multi-class weather image classification** (Cloudy, Rain, Shine, Sunrise) using **VGG16 transfer learning** on TensorFlow. The system includes a FastAPI backend, a real-time dashboard UI, Docker deployment, and Locust load testing.

---

## 📁 Directory Structure
```
Summative-assignment---MLOP/
├── README.md
├── api.py                  # FastAPI backend
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── locustfile.py           # Locust load testing
├── requirements.txt
├── notebook/
│   └── MY_SUMMATIVE.ipynb  # Full EDA + training notebook
├── src/
│   ├── preprocessing.py    # Data pipeline
│   ├── model.py            # Model build / train / retrain
│   └── prediction.py       # Inference utilities
├── data/
│   ├── train/              # Training images (per class)
│   └── test/               # Test images (per class)
├── models/
│   └── weather_model.h5    # Saved model (generated after training)
└── ui/
    └── templates/
        └── index.html      # Dashboard UI
```

---

## ⚙️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/Summative-assignment---MLOP.git
cd Summative-assignment---MLOP
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the model
The model is hosted on GitHub Releases (too large for the repo):
```bash
curl -L https://github.com/Innocent-Gershon/Summative-assignment---MLOP/releases/download/v1.0.0/weather_model_final.h5 \
     -o models/weather_model_final.h5
```
Or download manually from the [Releases page](https://github.com/Innocent-Gershon/Summative-assignment---MLOP/releases) and place it in the `models/` folder.

### 4. Prepare dataset
Download the [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) and place images in:
```
data/train/Cloudy/   data/train/Rain/   data/train/Shine/   data/train/Sunrise/
data/test/Cloudy/    data/test/Rain/    data/test/Shine/    data/test/Sunrise/
```
Or use the notebook to auto-split from the raw archive.

### 4. Run the API and UI
```bash
# Terminal 1 — FastAPI backend
python api.py
# → http://localhost:8000/docs

# Terminal 2 — Streamlit UI
streamlit run app.py
# → http://localhost:8501
```

### 5. Train the model
```bash
# Via UI: click "Train / Retrain" tab → Start Training
# Or via API:
curl -X POST "http://localhost:8000/train?epochs=25"
```

---

## 🐳 Docker Deployment

### Single container
```bash
docker build -t weather-classifier .
docker run -p 8000:8000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models weather-classifier
```

### Multi-container with Nginx (load balancing)
```bash
docker-compose up --scale weather-api=3
# API available at http://localhost:80
```

---

## 🦗 Load Testing with Locust

### Interactive UI
```bash
locust -f locustfile.py --host http://localhost:8000
# Open http://localhost:8089
```

### Headless (CLI) — record latency results
```bash
locust -f locustfile.py --host http://localhost:8000 \
       --headless -u 50 -r 5 --run-time 60s --html locust_report_50users.html

locust -f locustfile.py --host http://localhost:8000 \
       --headless -u 200 -r 20 --run-time 60s --html locust_report_200users.html
```
Results (latency & RPS) are saved to the HTML reports.

---

## 🔁 Retraining Trigger
1. Upload new labelled images via **Upload Data** tab (or `POST /upload`)
2. Click **Trigger Retraining** in the **Train / Retrain** tab (or `POST /retrain`)
3. The model fine-tunes on the updated dataset in the background
4. Status is polled automatically; model reloads on completion

---

## 📊 Flood Request Results
| Containers | Users | Avg Latency (ms) | RPS |
|-----------|-------|-----------------|-----|
| 1         | 50    | ~TBD            | ~TBD |
| 1         | 200   | ~TBD            | ~TBD |
| 3         | 200   | ~TBD            | ~TBD |

*(Fill in after running Locust experiments)*

---

## 📓 Notebook
`notebook/MY_SUMMATIVE.ipynb` contains:
- Data acquisition & EDA
- 3 feature visualizations with interpretations
- Model training (VGG16 transfer learning)
- Evaluation: accuracy, precision, recall, F1, confusion matrix, ROC-AUC
- Prediction function demo

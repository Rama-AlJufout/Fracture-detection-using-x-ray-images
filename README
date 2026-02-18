# 🦴 Fracture Detection Using X-Ray Images

A deep learning web application that detects bone fractures in X-ray images using a CNN model with Grad-CAM visualization to highlight the fracture region.

---

## 📌 Overview

This project uses a pre-trained ResNet-based model fine-tuned for binary classification:
- **Fracture** — fracture detected in the X-ray
- **No Fracture** — no fracture detected

When a fracture is detected, the app overlays a **Grad-CAM heatmap** and draws an **arrow pointing to the strongest activation region**, helping localize the suspected fracture area visually.

---

## 🚀 Features

- Upload an X-ray image via a simple web interface
- Get an instant prediction with confidence score
- Visual fracture localization using Grad-CAM + arrow annotation
- REST API built with FastAPI
- Deployable on Render

---

## 🗂️ Project Structure

```
Fracture-detection-using-x-ray-images/
│
├── models/
│   └── fracture_model.pth       # Trained model weights
│
├── templates/
│   └── index.html               # Frontend HTML page
│
├── arrow_utils.py               # Arrow drawing utilities
├── gradcam.py                   # Grad-CAM implementation
├── model.py                     # Model architecture (ResNet-based)
├── main.py                      # FastAPI app entry point
└── requirements.txt             # Python dependencies
```

---

## 🧠 Model

- **Architecture:** ResNet (custom head for binary classification)
- **Input size:** 224×224 RGB
- **Output:** Fracture / No Fracture + confidence score
- **Visualization:** Grad-CAM on `layer4` to generate class activation maps

---

## 🖥️ API Endpoints

| Method | Endpoint    | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/`         | Serves the web UI                        |
| POST   | `/predict`  | Accepts an image file, returns prediction |

### `/predict` Response Example

```json
{
  "prediction": "Fracture",
  "confidence": "94.3%",
  "image_base64": "<base64-encoded-annotated-image>"
}
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Fracture-detection-using-x-ray-images.git
cd Fracture-detection-using-x-ray-images
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the model weights

Place your trained model file at:
```
models/fracture_model.pth
```

### 4. Run the app

```bash
uvicorn main:app --reload
```

Then open your browser at [http://localhost:8000](http://localhost:8000)

---

## ☁️ Deployment

This project is configured for deployment on **[Render](https://render.com)**. Make sure your `requirements.txt` is up to date and your start command is:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 📦 Requirements

Key dependencies include:

- `fastapi`
- `uvicorn`
- `torch` + `torchvision`
- `opencv-python`
- `Pillow`
- `numpy`

See `requirements.txt` for the full list.

---

## 📄 License

This project is for educational and research purposes.

## 👩‍💻 Author

Developed as part of an AI medical imaging project By Rama Al-Jufout.
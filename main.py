from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import shutil
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from model import get_model
import uuid

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Fracture Detection API")

app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

templates = Jinja2Templates(directory="templates")

model = get_model().to(DEVICE)
model.load_state_dict(torch.load("models/fracture_model.pth", map_location=DEVICE))
model.eval()


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients[0].detach().cpu().numpy()
        acts = self.activations[0].detach().cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE))
        cam = cam / (cam.max() + 1e-8)
        return cam


cam_generator = GradCAM(model, model.layer4)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_strongest_point(cam):
    y, x = np.unravel_index(np.argmax(cam), cam.shape)
    return int(x), int(y)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse(status_code=400, content={"error": "Please upload an image"})

    filename = f"result_{uuid.uuid4().hex}.jpg"
    OUTPUT_IMAGE = os.path.join(UPLOAD_DIR, filename)

    image_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

    output = model(input_tensor)
    class_idx = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][class_idx].item()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    label = "Fracture" if class_idx == 0 else "No Fracture"

    if class_idx == 0:
        cam = cam_generator.generate(input_tensor, class_idx)
        x, y = get_strongest_point(cam)

        arrow_start = (max(x - 80, 0), max(y - 80, 0))
        arrow_end = (x, y)

        cv2.arrowedLine(
            img,
            arrow_start,
            arrow_end,
            color=(255, 255, 255),
            thickness=3,
            tipLength=0.25
        )

    cv2.imwrite(OUTPUT_IMAGE, img)

    return JSONResponse({
        "image_url": f"/static/{filename}",
        "prediction": label,
        "confidence": f"{confidence*100:.1f}%"
    })

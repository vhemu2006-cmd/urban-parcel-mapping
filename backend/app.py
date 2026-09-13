from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import os
import sys
import base64
import cv2
import numpy as np
import torch

# Reduce PyTorch memory usage
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

from ai_model.model import model
from backend.agent import generate_land_use_report


app = FastAPI(
    title="Urban Parcel Mapping API",
    description="AI-based urban land-use segmentation using U-Net",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

device = torch.device("cpu")
print("Using device:", device)

# Load model on CPU
MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "best_unet.pth"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = model.to(device)

state_dict = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

# Disable gradients permanently for inference
for parameter in model.parameters():
    parameter.requires_grad = False

print("Trained model loaded successfully")

CLASS_NAMES = {
    0: "No-data",
    1: "Background",
    2: "Building",
    3: "Road",
    4: "Water",
    5: "Barren Land",
    6: "Forest",
    7: "Agriculture"
}

COLOR_MAP = {
    0: (0, 0, 0),
    1: (255, 255, 255),
    2: (255, 0, 0),
    3: (0, 0, 255),
    4: (0, 255, 255),
    5: (255, 255, 0),
    6: (0, 255, 0),
    7: (0, 128, 0)
}


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Urban Parcel Mapping API is running"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "U-Net",
        "encoder": "ResNet18",
        "dataset": "LoveDA Urban",
        "classes": list(CLASS_NAMES.values()),
        "device": "cpu"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        contents = await file.read()

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return {
                "status": "error",
                "message": "Invalid image file"
            }

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        resized_image = cv2.resize(
            image_rgb,
            (256, 256),
            interpolation=cv2.INTER_AREA
        )

        # Convert without NumPy-Torch bridge
        image_tensor = torch.frombuffer(
            resized_image.tobytes(),
            dtype=torch.uint8
        ).clone()

        image_tensor = image_tensor.reshape(
            256,
            256,
            3
        )

        image_tensor = image_tensor.permute(
            2,
            0,
            1
        ).float()

        image_tensor = image_tensor.div_(255.0)
        image_tensor = image_tensor.unsqueeze(0)

        # Inference mode uses less memory
        with torch.inference_mode():
            output = model(image_tensor)
            prediction_tensor = torch.argmax(
                output,
                dim=1
            ).squeeze(0)

        prediction = np.array(
            prediction_tensor.tolist(),
            dtype=np.int64
        )

        del output
        del prediction_tensor
        del image_tensor

        total_pixels = prediction.size

        land_use_analysis = {}

        for class_id, class_name in CLASS_NAMES.items():

            pixel_count = int(
                np.sum(prediction == class_id)
            )

            percentage = round(
                (pixel_count / total_pixels) * 100,
                2
            )

            land_use_analysis[class_name] = {
                "pixels": pixel_count,
                "percentage": percentage
            }

        colored_mask = np.zeros(
            (256, 256, 3),
            dtype=np.uint8
        )

        for class_id, color in COLOR_MAP.items():
            colored_mask[prediction == class_id] = color

        _, original_buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(
                resized_image,
                cv2.COLOR_RGB2BGR
            )
        )

        original_base64 = base64.b64encode(
            original_buffer.tobytes()
        ).decode("utf-8")

        _, mask_buffer = cv2.imencode(
            ".png",
            cv2.cvtColor(
                colored_mask,
                cv2.COLOR_RGB2BGR
            )
        )

        segmentation_base64 = base64.b64encode(
            mask_buffer.tobytes()
        ).decode("utf-8")

        try:
            ai_report = generate_land_use_report(
                land_use_analysis
            )
        except Exception as agent_error:
            ai_report = {
                "summary": "Land-use analysis completed successfully.",
                "recommendations": [
                    "Review the predicted land-use distribution.",
                    "Use additional geographic data for detailed planning."
                ],
                "agent_error": str(agent_error)
            }

        return {
            "status": "success",
            "filename": file.filename,
            "image_size": {
                "width": 256,
                "height": 256
            },
            "predicted_classes": [
                int(class_id)
                for class_id in np.unique(prediction)
            ],
            "land_use_analysis": land_use_analysis,
            "ai_report": ai_report,
            "original_image": original_base64,
            "segmentation_image": segmentation_base64
        }

    except Exception as error:
        print("Prediction error:", str(error))

        return {
            "status": "error",
            "message": str(error)
        }
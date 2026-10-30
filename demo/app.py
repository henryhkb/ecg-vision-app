# ECG-Vision demo: detects possible atrial fibrillation (AF) from an image of a 12-lead ECG.
# Research and learning project, not a medical device.

# gradio: builds the web page | torch/torchvision: run the trained ResNet-18 | PIL: opens images
import gradio as gr
import spaces  # Hugging Face ZeroGPU: lends the app a GPU only while a prediction runs
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Threshold chosen on the validation set (see the GitHub README): probability >= 0.45 means "possible AF"
THRESHOLD = 0.45

# Rebuild the same ResNet-18 used in training (1 output = the AF score) and load the trained weights
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 1)
model.load_state_dict(torch.load("resnet18_augmented_best.pt", map_location="cpu"))
model.eval()

# Same image preparation as in training: resize, convert to numbers, rescale like ImageNet
prepare = transforms.Compose([
    transforms.Resize((240, 496)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@spaces.GPU(duration=20)  # ask ZeroGPU for a GPU for up to 20 seconds per prediction
def predict(image):
    """Return the model's AF probability for one ECG image, and a short explanation."""
    if image is None:
        return None, "Please upload or take a photo of a 12-lead ECG first."
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    x = prepare(image.convert("RGB")).unsqueeze(0).to(device)   # one image -> a batch of 1
    with torch.no_grad():                                        # predicting only, no learning
        p_af = torch.sigmoid(model(x)).item()                    # AF probability between 0 and 1
    if p_af >= THRESHOLD:
        verdict = f"**Possible atrial fibrillation** (AF probability {p_af:.2f}, threshold {THRESHOLD})."
    else:
        verdict = f"**No AF detected** (AF probability {p_af:.2f}, threshold {THRESHOLD})."
    note = ("\n\nThis is a research demo. It only distinguishes normal sinus rhythm from AF, "
            "so other heart conditions are not detected. Always have an ECG read by a qualified clinician.")
    return {"Possible AF": p_af, "Normal": 1 - p_af}, verdict + note


with gr.Blocks(title="ECG-Vision") as demo:
    gr.Markdown(
        "# ECG-Vision: atrial fibrillation from ECG images\n"
        "Upload a photo or image of a **standard 12-lead ECG** (3 rows of 4 leads plus a lead II rhythm strip). "
        "For best results, crop the photo to the ECG itself and keep it reasonably straight and well lit.\n\n"
        "⚠️ **Research demo only. Not a medical device. Do not use it for diagnosis or treatment decisions.** "
        "Uploaded images are only used to make the prediction and are not saved by this app."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(type="pil", label="ECG image")
            button = gr.Button("Analyse ECG", variant="primary")
            # Two example iPhone photos of PTB-XL test ECGs (one AF, one normal), so visitors can try it without their own ECG
            gr.Examples(examples=[["IMG_7402.JPG"], ["IMG_7399.JPG"]],
                        inputs=image_in, label="No ECG to hand? Click an example, then press Analyse ECG")
        with gr.Column():
            probs_out = gr.Label(num_top_classes=2, label="Model output")
            text_out = gr.Markdown()
    button.click(predict, inputs=image_in, outputs=[probs_out, text_out])
    gr.Markdown(
        "Model: ResNet-18 fine-tuned on 9,718 ECGs from the PTB-XL dataset, with photo-like augmentation. "
        "Test F1 0.954 on clean images and 0.92–0.96 on simulated blur, tilt, perspective and lighting changes. "
        "Code and full write-up: [github.com/henryhkb/ecg-vision-app](https://github.com/henryhkb/ecg-vision-app)"
    )

demo.launch()

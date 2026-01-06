import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from model import get_model
import torchvision.transforms as transforms


@st.cache_resource
def load_model():
    model = get_model()
    model.load_state_dict(torch.load("my_nn_model.pth", map_location = "cpu"))
    model.eval()
    return model

model = load_model()

classes = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Drawing Classifier")
st.write("Draw an object and click **Predict**")

canvas = st_canvas(
    stroke_width = 11,
    stroke_color = "white",
    background_color = "black",
    width = 280,
    height = 280,
    drawing_mode = "freedraw",
    key = "canvas"
)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess(img):
    img = Image.fromarray(img.astype("uint8"))
    img = img.convert("RGB")

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    img = transform(img)

    return img.unsqueeze(0)


if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data[:,:,:3]
        input_tensor = preprocess(img)

        with torch.no_grad():
            logits = model(input_tensor)
            T = 2.0
            probs = torch.softmax(logits/T, dim = 1)[0]

        st.subheader("Prediction Prob : ")

        for i, p in enumerate(probs):
            st.write(f"**{classes[i]}** : {p.item()*100:.2f}%")
        
        st.success(f"Final Prediction: **{classes[torch.argmax(probs)]}**")
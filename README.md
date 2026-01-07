# 🧥 DrawIt – Fashion‑MNIST Sketch Classifier (VGG16)

DrawIt is an interactive **deep‑learning deployment project** where users can **freehand‑draw fashion items** on a canvas and get **real‑time predictions** using a **VGG16‑based CNN** trained on the **Fashion‑MNIST dataset**.

This project demonstrates:
- Transfer Learning with **VGG16**
- Correct **training vs inference preprocessing alignment**
- Real‑time inference using **Streamlit**
- Handling **out‑of‑distribution (OOD)** input

---

## 🚀 Demo Overview

1. User draws a fashion item (shoe, shirt, bag, etc.) on a canvas
2. The drawing is preprocessed to match training conditions
3. A fine‑tuned **VGG16 model** predicts probabilities for all 10 Fashion‑MNIST classes
4. The app displays:
   - Class‑wise confidence scores
   - Final predicted label

---

## 🧠 Model Details

- **Base Model:** VGG16 (ImageNet pretrained)
- **Modified Classifier:**

```python
nn.Sequential(
    nn.Linear(25088, 1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 10)
)
```

- **Loss Function:** Cross‑Entropy Loss
- **Optimizer:** Adam
- **Training Accuracy:** ~92.8%

The model weights are saved using:
```python
torch.save(model.state_dict(), "model.pth")
```


---

## 🔄 Training Preprocessing (IMPORTANT)

The model was trained using the following transforms:

```python
transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

⚠️ **Inference preprocessing strictly matches training preprocessing** to avoid distribution shift and over‑confident predictions.

---

## 🖌️ Inference Pipeline

During inference:
1. Canvas drawing is converted to RGB
2. Empty borders are cropped (bounding‑box crop)
3. Training‑time transforms are applied
4. Image is reshaped to `(1, 3, 224, 224)`
5. Softmax probabilities are computed

Temperature scaling and confidence thresholds are used to reduce overconfidence.

---

## 📦 Project Structure

```
DrawIt/
│
├── app.py          # Streamlit application
├── model.py        # VGG16 model definition
├── model.pth       # Trained weights
├── README.md       # Project documentation
```

---

## 🛠 Installation

### 1️⃣ Clone the repository
```bash
git clone <repo-url>
cd DrawIt
```

### 2️⃣ Install dependencies
```bash
pip install torch torchvision streamlit streamlit-drawable-canvas pillow numpy
```

---

## ▶️ Run the Application

```bash
python -m streamlit run app.py
```

Open your browser at:
```
http://localhost:8501
```

---

## 🎯 Fashion‑MNIST Classes

```text
0 → T‑shirt/top
1 → Trouser
2 → Pullover
3 → Dress
4 → Coat
5 → Sandal
6 → Shirt
7 → Sneaker
8 → Bag
9 → Ankle boot
```
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/d73f14d5-5232-4507-87b2-aeb892577057" />
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/acf05faf-2a1c-4e71-a49e-3c90f052b872" />

---

## ⚠️ Known Limitations

- Model was trained on **photographic images**, not sketches
- Freehand drawings are **out‑of‑distribution** inputs
- Softmax can produce **high confidence for incorrect predictions**

Mitigations implemented:
- Exact preprocessing match
- Border cropping
- Temperature scaling
- Confidence thresholding

---


## 📜 License

This project is for **educational purposes**.

---

### ⭐ If you found this project useful, consider giving it a star!


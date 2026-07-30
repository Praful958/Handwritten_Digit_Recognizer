# ✍️ AI Handwritten Digit Recognizer

A Deep Learning based Handwritten Digit Recognition System built using **PyTorch CNN** and **Streamlit**.  
The application allows users to draw handwritten digits and get real-time predictions with confidence scores.

---

## 🚀 Features

✅ CNN based image classification using PyTorch  
✅ Real-time handwritten digit prediction  
✅ Streamlit interactive web interface  
✅ Top-3 prediction results with confidence score  
✅ Image preprocessing pipeline  
✅ Model evaluation system  
✅ Training history tracking  
✅ Accuracy and loss visualization  
✅ Confusion matrix analysis  

---

## 🛠️ Tech Stack

### Deep Learning
- PyTorch
- Torchvision
- Convolutional Neural Network (CNN)

### Data Processing
- NumPy
- Pillow

### Visualization
- Matplotlib
- Scikit-learn

### Deployment
- Streamlit

---

## 📂 Project Architecture

```
Handwritten_Digit_Recognizer/

│
├── app.py                  # Streamlit Web Application
├── train.py                # Training Pipeline
├── evaluate.py             # Model Evaluation
├── confusion_matrix.py     # Confusion Matrix Generator
├── plot_history.py         # Training Graph Generator
├── config.py               # Configuration Management
├── requirements.txt
├── README.md
│
├── data/
│   └── MNIST Dataset
│
├── models/
│   └── best_model.pth      # Trained CNN Model
│
├── outputs/
│   ├── history.json
│   ├── loss_graph.png
│   ├── accuracy_graph.png
│   └── confusion_matrix.png
│
└── src/
    ├── model.py            # CNN Architecture
    ├── dataset.py          # Data Loading
    ├── trainer.py           # Training Logic
    ├── predictor.py         # Prediction Logic
    ├── preprocess.py        # Image Processing
    └── evaluation.py        # Evaluation Logic
```

---

## 🧠 Model Architecture

The model uses a Convolutional Neural Network:

```
Input Image (28x28)

        ↓

Convolution Layer

        ↓

ReLU Activation

        ↓

Max Pooling

        ↓

Convolution Layer

        ↓

Fully Connected Layers

        ↓

10 Class Output (0-9)
```

---

## 📊 Model Performance

Dataset:
- MNIST Handwritten Digits

Training Accuracy:
- 99%+

Evaluation Metrics:
- Accuracy
- Confusion Matrix
- Loss Curve

---

## ⚙️ Installation

Clone repository:

```bash
git clone <repository-url>
```

Go inside project:

```bash
cd Handwritten_Digit_Recognizer
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Project

### Train Model

```bash
python train.py
```

### Evaluate Model

```bash
python evaluate.py
```

### Generate Graphs

```bash
python plot_history.py
```

### Start Web Application

```bash
streamlit run app.py
```

---

## 📸 Application Preview

(Add Streamlit screenshots here)

---

## 🔮 Future Improvements

- Better image centering algorithm
- Data augmentation
- Advanced CNN architecture
- Cloud deployment
- Mobile application integration

---

## 👨‍💻 Author

**Patil Praful Sajan**

AI/ML Engineer Aspirant

---

⭐ If you like this project, consider giving it a star.
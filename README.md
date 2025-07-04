# 🧪 Ingredient Health Analyzer

This project is a Streamlit web application that helps users analyze the healthiness of food products by extracting and evaluating the ingredient list from a photo of a food package. It uses a custom-trained YOLO model to detect the ingredient box, EasyOCR to extract text, and a local LLM (Ollama + Mistral) to provide a health analysis of the ingredients.

---

## 🚀 Features

- **Upload Image:** Upload a photo of a food package.
- **Detect Ingredient Box:** Uses YOLO to locate and crop the ingredient list on the packaging.
- **OCR Extraction:** Extracts the ingredient text with EasyOCR.
- **LLM Health Analysis:** Analyzes the extracted ingredients using a local LLM and provides:
  - Healthiness verdict
  - Harmful ingredients highlighted
  - Potential long-term health impacts
  - Suggestions for healthier alternatives

---

## 🖼️ Demo

1. **Upload a food package image**
2. The model detects and extracts the **ingredient list**
3. The local LLM analyzes health impacts and returns suggestions

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ShivaGujja/food-inggredient-analysis.git
cd food-inggredient-analysis
```

### 2. Install dependencies

It's recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- `streamlit`
- `opencv-python`
- `numpy`
- `ultralytics`
- `easyocr`
- `ollama` (Python package for local LLMs)

### 3. Download/Place the YOLO Model

Ensure you have a YOLOv8 `.pt` model trained to detect ingredient boxes.  
Update `MODEL_PATH` in `app.py` with the path to your model weights.

### 4. Install & Run Ollama

You need a local Ollama server running the `mistral` model (or similar).  
See [Ollama documentation](https://ollama.com/) or run:

```bash
ollama serve
ollama pull mistral
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

- **MODEL_PATH:**  
  Set the path to your custom YOLO weights in `app.py`:
  ```python
  MODEL_PATH = r"path\to\your\best.pt"
  ```
- **LLM Model:**  
  The code uses `mistral` by default. You can change this in the `call_local_llm` function.

---

## 📂 File Structure

```
├── app.py              # Main Streamlit app
├── README.md           # This file
├── requirements.txt    # Python dependencies
└── ...                 # (Other files, model weights, etc.)
```

---

## 📝 Notes

- The quality of detection and OCR depends on the clarity of the uploaded image and the accuracy of the trained YOLO model.
- The LLM analysis is only as good as the model you use locally with Ollama.
- All processing is done locally — no data leaves your machine (privacy-friendly).

---

## 🧑‍💻 Author

**Shiva Gujja**  
[GitHub: ShivaGujja](https://github.com/ShivaGujja)

---

## 📜 License

This project is for educational/research purposes. Please check the licenses of the datasets and models you use.

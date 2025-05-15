# 🧠 Surgical Image Dataset JSON Converter (CholecT45)

This repository provides a robust tool for converting raw laparoscopic surgical image datasets into a clean, structured JSON format. It was designed for use with the CholecT45 dataset, which contains over 100,000+ frames annotated for instruments, verbs, and anatomical targets.

---

## 🛉 Why This Exists

Deep learning workflows in medical AI often suffer from:

* ❌ Inconsistent data formats
* ❌ Poor label accessibility
* ❌ Scattered annotations

During my work on multi-label action recognition in laparoscopic surgeries, I faced the challenge of efficiently handling large-scale annotated image datasets. Converting and structuring them manually was error-prone, slow, and incompatible with modern pipelines.

> This tool solves that by transforming raw surgical frames and text-based labels into a single, JSON-based format that is clean, portable, and ML-ready.

---

## ✅ Key Features

* Converts raw `.png` image sequences into base64-encoded JSON entries
* Includes multi-label annotations for instruments, verbs, and targets
* Adds both numeric labels and human-readable names
* Supports train/val/test splits from official cross-validation folds
* Easily integrable into PyTorch or TensorFlow pipelines

---

## 💡 Use Case

This tool is useful for:

* Medical AI research projects
* Surgical action detection or instrument segmentation
* Transformer-based models requiring JSON-based inputs (e.g., LLaVA, BLIP, or CLIP variants)
* Efficient training on cloud (e.g., Google Colab, HuggingFace datasets)

---

## ⚒️ How It Works

### Input:

* Raw image folder structure like:

  ```
  CholecT45/
  ├── data/
  │   └── VID01/
  │       └── 000000.png
  ├── instrument/
  │   └── VID01.txt
  ├── verb/
  └── target/
  ```

* Label `.txt` files with format: `frame_number, label_1, label_2, ...`

### Output:

* Structured JSON file like:

```json
{
  "video_no": "VID01",
  "image_no": "000123",
  "image": "<base64 PNG>",
  "instrument_labels": [0, 1, 0, 0, 0, 0],
  "instrument": "grasper, bipolar",
  "verb_labels": [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
  "verb": "dissect, null_verb",
  "target_labels": [...],
  "target": "cystic_plate"
}
```

---

## 📆 Files Created

* `ivt_train.json`
* `ivt_val.json`
* `ivt_test.json`

Each contains thousands of samples structured as shown above.

---

## 🚀 How to Use

```bash
python convert_to_json.py
```

Make sure to update `dataset_dir` at the bottom of the script to point to your local CholecT45 folder.

---

## 📁 Repo Structure

```
.
├── convert_to_json.py    # Main script to run conversion
├── output/
│   ├── ivt_train.json
│   ├── ivt_val.json
│   └── ivt_test.json
├── utils/
│   ├── label_maps.py     # (optional) if modularized
│   └── image_utils.py    # (optional) if modularized
├── README.md
```

---

## ✍️ Author

**Davang Sikand**
Research Fellow – Computer Vision for Surgery
Plaksha University | Fortis Hospitals | Former Publicis Sapient
[LinkedIn](https://www.linkedin.com/in/frustratedmonk) | [GitHub](https://github.com/frustratedmonk)

---

## 📌 Final Note

If you're working with surgical datasets at scale and want clean, reliable data pipelines, this tool is for you. It’s optimized for scalability, interpretability, and training readiness — something I wish I had when I began my project.

Feel free to fork, adapt, or reach out for collaborations.


# Ideathon Team BATMEN  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  
[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](https://example.com)  

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

---

## 🚀 Project Overview
This project is developed by **Team BATMEN** as part of the **Ideathon** challenge. The objective is to build an **AI-powered welding defect detection system** using **YOLOv8**. The solution automates the detection and classification of welding defects using deep learning, significantly improving quality control in manufacturing.

---

## 🌟 Key Features
- **Welding Defect Detection:** Detects and classifies different types of welding defects using YOLOv8.
- **Real-Time Analysis:** Processes images and videos in real-time for defect detection.
- **Model Training & Fine-Tuning:** Includes scripts for training and improving the model.
- **Dataset Management:** Organized dataset structure with labeled images for efficient training.
- **Scalable & Modular:** Designed for easy integration into industrial pipelines.

---

## 🛠️ Technologies Used
- **Deep Learning:** YOLOv8 for object detection
- **Programming Language:** Python
- **Frameworks & Libraries:** PyTorch, OpenCV, NumPy, Pandas, Matplotlib
- **Tools:** Jupyter Notebook, GitHub,

---

## 📁 Directory Structure
```
Ideathon_team_BATMEN/
├── data/
│   ├── Weld/
│   ├── runs/detect/
│   ├── welding_defects/
│   ├── custom_data.yaml
│   ├── yolo11n.pt
│   └── yolov8n.pt
├── test/
│   ├── images/
│   ├── labels/
│   ├── train/
│   └── valid/
├── runs/detect/
├── welding_defects/
├── custom_data.yaml
├── yolo11n.pt
├── yolov8n.pt
├── training.py
├── improve_training.py
├── video.py
├── LICENSE
└── README.md
```

### Data Directory
- **Weld/**: Contains welding-related image datasets.
- **runs/detect/**: Stores detection results.
- **welding_defects/**: Includes datasets related to welding defects.
- **custom_data.yaml**: Configuration file for defining dataset structure.
- **yolo11n.pt** & **yolov8n.pt**: Pre-trained YOLO model weights.

### Training Scripts
- **training.py**: Script for training YOLOv8 on custom dataset.
- **improve_training.py**: Fine-tuning script to enhance model accuracy.
- **video.py**: Runs real-time inference on video data.

---

## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Sengathir24/Ideathon_team_BATMEN.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Ideathon_team_BATMEN
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

---

## 🏃‍♂️ Usage

### 1️⃣ Train the Model
To train the YOLOv8 model on your dataset:
```bash
python training.py
```

### 2️⃣ Run Inference on Images
To detect welding defects on test images:
```bash
python video.py --weights yolo8n.pt --source data/Weld/test/images
```

### 3️⃣ Improve Training
Fine-tune the model with:
```bash
python improve_training.py
```

---

## 🤝 Contributing
We welcome contributions! Follow these steps to contribute:
1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and test thoroughly
4. Commit changes: `git commit -m 'Add new feature'`
5. Push to the branch: `git push origin feature/your-feature`
6. Open a pull request

---

## 📝 License
This project is licensed under the [MIT License](LICENSE).

---

## 📧 Contact
**Team BATMEN**  
- **Lead Developer:** Sengathirsoorian  
- **GitHub:** [@Sengathir24](https://github.com/Sengathir24)  
- **Email:** [sengathirsoorian@gmail.com](sengathirsoorian@gmail.com)  


---

## 🙏 Acknowledgments
- Special thanks to **Coding club** for organizing the Ideathon.
- **YOLOv8 community** for making object detection accessible.
- **Mentors & Teammates** for support and collaboration.
```


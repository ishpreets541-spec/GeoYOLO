# 🌿 GeoYOLO
**Geospatial Machine Learning Engine for Vegetation Segmentation & Fractional Cover**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8-blue?logo=ultralytics&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

**GeoYOLO** is an end-to-end geospatial computer vision pipeline designed to automate vegetation mapping. By combining automated NDVI-based dataset annotation, YOLOv8 instance segmentation, and an interactive frontend, this tool extracts fractional vegetation cover directly from multispectral satellite imagery.

---

## 🚀 Live Demo
Access the live deployment on Streamlit Community Cloud: **[Insert Link Here]**

---

## 🧠 Core Features

*   **Multispectral Image Support:** Natively processes 13-band Sentinel-2 TIF files alongside standard RGB imagery (JPG, PNG)[cite: 1, 2].
*   **Automated Dataset Annotation:** Automatically generates YOLO segmentation labels by calculating NDVI from Red (Band 4) and NIR (Band 8) channels, utilizing a >0.4 threshold to isolate vegetation boundaries[cite: 2].
*   **Fallback Masking:** Implements HSV color-space masking as an automated fallback for standard 3-band RGB imagery to ensure continuous annotation[cite: 2].
*   **Deep Learning Segmentation:** Leverages the Ultralytics YOLOv8 nano instance segmentation architecture (`yolov8n-seg.pt`), optimized via PyTorch and CUDA for high-speed feature extraction[cite: 3].
*   **Fractional Cover Analytics:** Dynamically merges individual YOLO detection masks into a unified array to calculate and output the exact percentage of vegetation cover across the analyzed spatial extent[cite: 1].
*   **Interactive Dashboard:** Features a Streamlit web interface providing side-by-side visual comparisons of original satellite inputs and the resulting AI-generated polygon overlays[cite: 1].

---

## ⚙️ Pipeline Architecture

The system operates across three primary modular scripts:

1.  **Auto-Annotator:** Scans image directories, applies spatial masking, extracts contours via OpenCV, and converts geometric boundaries into normalized YOLO polygon coordinates[cite: 2].
2.  **Model Training:** Configures the training loop for the YOLOv8 model using an image size of 640 and a batch size of 16, complete with early stopping (patience of 10) to optimize learning[cite: 3].
3.  **Inference Engine:** An application that caches the trained weights (`best.pt`), normalizes input tensors, executes predictions with high-resolution retina masks, and renders the analytics to the user[cite: 1].

---

## 💻 Local Installation & Setup

To deploy GeoYOLO locally for development or dataset generation:

```bash
# Clone the repository
git clone [https://github.com/yourusername/GeoYOLO.git](https://github.com/yourusername/GeoYOLO.git)
cd GeoYOLO

# Install the required spatial and deep learning dependencies
pip install ultralytics torch torchvision streamlit opencv-python tifffile Pillow numpy
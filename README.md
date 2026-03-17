# Real-time-sign-language-Translator-A-Machine-Learning-based-approach
This project is a real-time computer vision based application that captures live webcam input
and translates hand sign language gestures into readable text.

The system uses a CNN-based deep learning model trained on sign language data and performs
real-time gesture prediction using OpenCV. The project is designed to improve accessibility
for speech- and hearing-impaired individuals by enabling seamless human–computer interaction.

🔹 Features
- Real-time webcam-based gesture recognition
- Converts hand gestures into text instantly
- CNN-based deep learning model
- Smooth and responsive UI for real-time usage
- Designed for accessibility and assistive technology use cases

🔹Model Details
- Designed and trained a **1.6M-parameter CNN architecture**  
- Supports **26-class American Sign Language (A–Z)** recognition  
- Trained on **100K+ images** with proper train-test split  
- Achieved **~99% validation accuracy** after incremental training  
- Optimized for **real-time inference (<80ms per frame)**  
- Implemented normalization (1/255 scaling) and dropout for generalization  

> Note: Due to real-time inference constraints, the model may occasionally misclassify visually similar gestures, which is expected behavior in live computer vision systems.


🔹 Technologies Used
- Python
- Machine Learning
- Convolutional Neural Networks (CNN)
- OpenCV
- NumPy
- Streamlit 
- Computer Vision


#🎥 Demo Video

> 🎬 Click below to view the real-time working demo of the application  

👉 [▶️ Watch Demo](https://drive.google.com/file/d/1-jT-8IImkSuowjeWHsAlo_YaR9IAIadJ/view?usp=drivesdk)




 #📸 Screenshots

### 🔹 User Interface
![UI]([screenshots/ui](https://github.com/AditiGaure/Real-time-sign-language-Translator-A-Machine-Learning-based-approach/blob/main/screenshots/Screenshot%202026-03-17%20171902.png))

### 🔹 Real-Time Prediction (Example 1)
![Prediction P]([screenshots/p.](https://github.com/AditiGaure/Real-time-sign-language-Translator-A-Machine-Learning-based-approach/blob/main/screenshots/Screenshot%202026-03-17%20171953.png))

### 🔹 Real-Time Prediction (Example 2)
![Prediction Q]([screenshots/q.](https://github.com/AditiGaure/Real-time-sign-language-Translator-A-Machine-Learning-based-approach/blob/main/screenshots/Screenshot%202026-03-17%20172049.png))

### 🔹 Real-Time Prediction (Example 3)
![Prediction Q](screenshots/q.png)




🔹 Project Structure

sign_language_polished_UI.py   -> Main real-time prediction script with UI

test_model_output.py          -> Script to test trained model predictions


🔹 How to Run

1. Clone the repository
2. Install required dependencies
3. Run sign_language_polished_UI.py
4. Allow webcam access

    
🔹 Note

“Due to file size and ownership considerations, the trained model file (.h5) is not included. The complete architecture, preprocessing, and inference code are provided.”

🔹 How to Run

1. Clone the repository
2. Install required dependencies
3. Run sign_language_polished_UI.py
4. Allow webcam access

 ---
🔹 Author

Aditi Gaure


🔹Disclaimer:

This project and its source code are original work developed solely by the author.
Unauthorized copying, redistribution, modification, or use of this code or its implementation—in full or in part—is strictly prohibited without explicit prior permission from the author.

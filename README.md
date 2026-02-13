# 🧠 Early Detection of Parkinson’s Disease  
### AI-Based Voice Analysis System  
---

## 📌 Project Overview  

Parkinson’s Disease (PD) is a progressive neurodegenerative disorder that affects motor control and speech. Early detection plays a crucial role in slowing disease progression and improving patient quality of life.

This project presents a **non-invasive, AI-powered voice analysis system** that detects Parkinson’s Disease using acoustic biomarkers extracted from sustained vowel phonation (`/a/` sound).

The system leverages:

- 🎙️ Voice Biomarkers  
- 📊 MDVP Acoustic Features  
- 🤖 Machine Learning (Random Forest Classifier)  
- 🌐 Web-Based User Interface  

---

## 🚀 Key Features  

- 🎤 Upload or record voice samples  
- 🔊 Automatic noise removal & silence trimming  
- 📈 Extraction of 22-dimensional MDVP feature set  
- 🌲 Random Forest classification model  
- 📊 Confidence score output  
- 💾 Secure result storage  
- 🌍 Remote screening capability  

---

## 🏗️ System Architecture  
<img width="595" height="728" alt="image" src="https://github.com/user-attachments/assets/2a8ec692-5430-4f77-80d7-08a56c363006" />

---

## ScreenShots
<img width="1280" height="607" alt="image" src="https://github.com/user-attachments/assets/586aa1c5-d8c1-4370-b3ab-4707e9e65df9" />
<img width="1280" height="612" alt="image" src="https://github.com/user-attachments/assets/302be268-92e3-43cf-bc4c-b09e5a8b3231" />
<img width="1280" height="614" alt="image" src="https://github.com/user-attachments/assets/93a6ff99-38d2-4d1c-b101-dbae8bcda010" />
<img width="1280" height="613" alt="image" src="https://github.com/user-attachments/assets/cb84badb-84ae-4525-b24f-34f8b26d8e6b" />
<img width="1280" height="608" alt="image" src="https://github.com/user-attachments/assets/e2ff175e-32cf-4712-94d5-bce4ae080bb2" />

---

## 🧮 Feature Extraction  

### 🎵 Pitch Features
- MDVP: Fo (Fundamental Frequency)  
- Fhi (Highest Frequency)  
- Flo (Lowest Frequency)  

### 📉 Jitter Metrics
- Jitter (%)  
- Jitter (Abs)  
- RAP  
- PPQ  
- DDP  

### 📊 Shimmer Metrics
- Shimmer  
- Shimmer (dB)  
- APQ3  
- APQ5  
- DDA  

### 🔇 Noise Measures
- HNR (Harmonics-to-Noise Ratio)  
- NHR (Noise-to-Harmonics Ratio)  

### 📈 Nonlinear Features
- RPDE  
- DFA  
- PPE  
- spread1  
- spread2  

---

## 🤖 Machine Learning Model  

- Algorithm: Random Forest Classifier  
- Feature Scaling: StandardScaler  
- Missing Value Handling: Median Imputation  
- Dataset: UCI Parkinson’s Voice Dataset  
- Evaluation Metrics:
  - Accuracy  
  - ROC Curve  
  - AUC Score  
  - Feature Importance  

---

## 🛠️ Flow Diagram
<img width="745" height="807" alt="image" src="https://github.com/user-attachments/assets/4adbca6f-67ab-4b6e-9859-0f969fda2460" />


### 💻 Programming Language
- Python 3.x  

### 📚 Libraries
- scikit-learn  
- librosa  
- pandas  
- numpy  
- matplotlib  
- joblib  
- parselmouth  
- flask  

### 🖥️ Development Tools
- Jupyter Notebook  
- VS Code / PyCharm  
- Anaconda / pip  

---

## 💾 Hardware Requirements  

- Intel i3/i5 Processor (minimum)  
- 8GB RAM  
- Microphone-enabled device  
- 500MB storage  

---

## 📊 Results  

The system successfully:

- Processes raw audio samples  
- Extracts clinically relevant voice biomarkers  
- Classifies as:
  - ✅ Healthy  
  - ⚠️ Parkinson’s Detected  
- Provides confidence percentage  

The model demonstrates strong potential for early-stage PD screening and remote healthcare support.

---

## 🔬 Testing Performed  

- Unit Testing  
- Integration Testing  
- Functional Testing  
- Performance Testing  
- Validation Testing  

---

## 🔮 Future Enhancements  

- 📱 Mobile Application Deployment  
- 🧠 Deep Learning Integration (CNN / Transformers)  
- 🌍 Multilingual Dataset Expansion  
- 🔍 Explainable AI (XAI) Visualization  
- 🏃 Multi-modal biomarker integration  

---

## 👨‍💻 Team Members  

- NAMAN A U
- AADITHYA A R   
- YADUNANDAN M N  
- KENISHA P 


---

## ⚠️ Disclaimer  

This project is intended for academic and research purposes only.  
It is not a certified medical diagnostic tool.  
Clinical decisions should always be made by qualified healthcare professionals.

---

## ⭐ Support  

If you find this project helpful:

- ⭐ Star this repository  
- 🍴 Fork it  
- 🤝 Contribute improvements  

---


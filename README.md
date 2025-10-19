# 🤖 Train and Deploy a Multimodal AI Model (2025)
### PyTorch · AWS · SageMaker · Next.js 15 · React · TailwindCSS

This project demonstrates how to **train, deploy, and integrate a multimodal AI model** using **PyTorch**, **AWS SageMaker**, and **Next.js 15**.  
The AI model processes **video, audio, and text inputs** to predict **emotion and sentiment**, and the trained model is deployed as an **API SaaS** application with a full web dashboard.

---

## 🚀 Overview

In this project, you’ll:
1. Train a multimodal AI model combining **text, video, and audio encoders**.
2. Implement **multimodal fusion** for sentiment & emotion classification.
3. Deploy the trained model using **AWS SageMaker Endpoints**.
4. Build a **SaaS dashboard** using **Next.js 15**, **React**, **TailwindCSS**, and **Auth.js**.
5. Manage **user authentication**, **API key access**, and **quota tracking**.
6. Visualize **real-time inference** and analysis results.

---

## 🧠 Features

- 🎥 **Video sentiment analysis**
- 🧩 **Multimodal fusion** of text, audio, and visual signals
- 🎙️ **Audio feature extraction**
- 📝 **Text embeddings** using **BERT**
- 📊 **Emotion and sentiment classification**
- ⚙️ **Model training, evaluation, and logging via TensorBoard**
- ☁️ **AWS S3** for video storage  
- 🤖 **SageMaker Endpoint** deployment for scalable inference  
- 🔐 **Auth.js authentication**
- 🔑 **API key & usage quota management**
- 📈 **Interactive dashboard** with real-time inference
- 🎨 **Modern UI built with Tailwind CSS**
- 💾 **Database schema for users and usage tracking**
- 🌍 **End-to-end integration from training to SaaS product**

---

## 🧰 Tech Stack

### 🧠 Machine Learning
- **PyTorch** — Model training and multimodal fusion  
- **TensorBoard** — Logging and performance visualization  
- **MELD Dataset** — For emotion and sentiment labels  
- **FFmpeg** — Video and audio preprocessing  

### ☁️ Cloud & Deployment
- **AWS S3** — Dataset and model storage  
- **AWS EC2** — Dataset preparation and instance management  
- **AWS SageMaker** — Training jobs and endpoint hosting  
- **IAM Roles** — Access control and API permissions  

### 💻 Web Application
- **Next.js 15** — Frontend framework  
- **React.js** — Component-based UI  
- **Tailwind CSS** — Styling  
- **Auth.js** — Authentication  
- **Prisma / PostgreSQL** — Database and ORM  
- **T3 Stack** — Full-stack foundation for the SaaS layer  

---

## 🧪 Dataset

**MELD (Multimodal EmotionLines Dataset)**  
A benchmark dataset for multimodal emotion recognition in conversations.  
🔗 [MELD Official Website](https://affective-meld.github.io/)

The model extracts:
- **Text features** via BERT  
- **Audio embeddings** via spectral analysis  
- **Visual features** via CNN encoders  
Then fuses them for **emotion classification** (`happy`, `sad`, `neutral`, `angry`, etc.)

---

## 📦 Folder Structure

```text
.
├── model/
│   ├── train.py               # Training script (PyTorch)
│   ├── model.py               # Multimodal fusion model
│   ├── dataset.py             # Custom dataset loader for MELD
│   └── inference.py           # Local inference
├── aws/
│   ├── s3_upload.py           # Upload dataset/model to S3
│   ├── sagemaker_train.py     # Training job creation
│   ├── sagemaker_deploy.py    # Endpoint deployment
│   └── invoke_endpoint.py     # Inference through API
├── web/
│   ├── app/                   # Next.js App Router
│   ├── components/            # UI components
│   ├── pages/api/             # API routes (Auth, Quotas, Inference)
│   └── styles/                # Tailwind styles
└── README.md
```

---

## ⚙️ Setup Instructions

### 🔧 Local Development
```bash
git clone https://github.com/yourusername/multimodal-ai-saas.git
cd multimodal-ai-saas
```

#### 1️⃣ Model Environment
```bash
cd model
pip install -r requirements.txt
python train.py
```

#### 2️⃣ AWS Configuration
```bash
aws configure
python aws/s3_upload.py
python aws/sagemaker_train.py
python aws/sagemaker_deploy.py
```

#### 3️⃣ SaaS Dashboard
```bash
cd web
npm install
npm run dev
```
Then open **http://localhost:3000**

---

## 💲 AWS Cost Overview

| Service | Cost Estimate | Notes |
|----------|----------------|-------|
| SageMaker Training | ~15 USD | One-time full training job |
| Endpoint Hosting | ~1.5 USD/hour | Pay per uptime hour |
| S3 Storage | Very low | Based on dataset/model size |
| IAM Roles, Users | Free | Used for permission setup |

💡 **Free-tier workaround:**  
- Skip dataset upload & EC2 — use provided model from Google Drive.  
- Use dummy data for the API demo (as shown in the tutorial).  
- Try AWS free-tier instances for experimentation.

---

## 🧩 Example Features Implemented
- Multimodal encoder fusion (text + audio + video)
- Real-time inference dashboard
- Authenticated API request system
- API key creation & quota tracking
- Deployment automation using SageMaker SDK

---

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/latest/dg/)
- [Next.js Docs](https://nextjs.org/docs)
- [Auth.js](https://authjs.dev/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Framer Motion](https://www.framer.com/motion/)

---

## 📜 License
This project is open-sourced under the **MIT License**.  
Feel free to fork, modify, or contribute responsibly.

---

© 2025 **Vishnu Vinayan**  
Built with ❤️ using **PyTorch**, **AWS**, and **Next.js 15**

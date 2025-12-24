# Semantic-Aware Adaptive Frequency Domain Filtering (SA-FDF)

**基于多模态语义感知的视频流自适应频域滤波与重建研究**

![Pipeline](assets/pipeline.png)

## 📖 Introduction (项目介绍)

In digital image and video processing, suppressing noise while preserving high-frequency edge details is a core challenge. Traditional linear smoothing filters (e.g., Gaussian, Mean filters) often blur edges while removing noise.

This project proposes a **hybrid architecture** combining Computer Vision (Semantic Segmentation) with Digital Signal Processing (Frequency Domain Filtering). By using **U-2-Net** to generate high-precision dynamic foreground masks, we apply differentiated frequency domain truncation strategies to the foreground and background.

**核心亮点：**

- [cite_start]**Hybrid Architecture:** Python (AI Vision) + MATLAB (DSP) 混合架构 [cite: 10, 56]。
- [cite_start]**Adaptive Filtering:** 对前景（人物）保留高频细节，对背景进行强力去噪，解决“去噪保边”的矛盾 [cite: 12]。
- [cite_start]**Significant Improvement:** PSNR 在静态图像中从 22.49dB 提升至 31.62dB [cite: 24]。

## 🛠️ Features (功能特性)

- [cite_start]**AI-Powered Masking:** 使用 `rembg` (U-2-Net) 自动提取高精度人物前景 [cite: 11]。
- [cite_start]**Dual-Channel Filtering:** * **Background:** Low cutoff frequency ($D_0 \approx 115.2$) for smooth "creamy" bokeh effect[cite: 86].
  - [cite_start]**Foreground:** High cutoff frequency ($D_0 \approx 384.0$) to preserve hair and facial details[cite: 87].
- [cite_start]**Soft Fusion:** Gaussian feathering applied to masks to ensure smooth transitions between regions[cite: 64].

## 📊 Results (实验结果)

### Visual Comparison

![Comparison](assets/comparison.png)
*Left: Noisy Input (PSNR 22.5dB) | [cite_start]Right: Smart Zone Filter (PSNR 31.6dB)* [cite: 108-109]

### Performance Analysis

![Performance](assets/report_chart.png)
The algorithm achieves a Global Gain of **+6.35 dB** in video streams, maintaining high SSIM (~0.8) without structural distortion[cite: 200, 204].

## 🚀 Quick Start (使用指南)

### Prerequisites

- MATLAB R2022a or later

- Python 3.8+
- Python Libraries: `opencv-python`, `rembg`, `numpy`

### Installation

1. Clone the repository:

   ```bash
   git clone [https://github.com/YourUsername/Semantic-Adaptive-Filter.git](https://github.com/YourUsername/Semantic-Adaptive-Filter.git)

Install Python dependencies:

Bash

pip install -r requirements.txt
Usage Workflow
Generate Masks (Python): Run the Python script to extract foreground masks from your video/image.

Bash

python src/python/video_mask.py
Apply DSP Filtering (MATLAB): Open src/matlab/main_video_process.m in MATLAB. Ensure the paths to the original video and generated mask are correct, then run the script.

📂 Project Structure
/src/python: AI segmentation scripts (U-2-Net based).

/src/matlab: Frequency domain filtering and reconstruction algorithms (FFT/IFFT).

/docs: Detailed project report (PDF).

📝 Citation
If you find this project useful, please cite:

Hui, T. (2025). Research on Adaptive Frequency Domain Filtering and Reconstruction of Video Streams Based on Multi-modal Semantic Perception. Journal of Xidian University (Student Project).

📄 License
This project is licensed under the MIT License.

---

# Handwritten Digit Recognition using CNN in MATLAB

This project implements a Convolutional Neural Network (CNN) to recognize and classify handwritten digits using the built-in MATLAB DigitDataset. The model is optimized for high accuracy and robustness against variations in handwriting styles.

---

## 🚀 Overview

The system follows a standard machine learning workflow and achieves strong classification performance through spatial feature extraction and regularization techniques.  
To improve generalization, the model incorporates **data augmentation** and **dropout layers**, reducing overfitting and enhancing performance on unseen data.

---

## 🛠 Methodology

The implementation is structured into the following key phases:

- **Dataset:** $28 \times 28$ grayscale images of digits (0–9)
- **Data Split:** 80% training, 20% testing
- **Data Augmentation:** Random rotations (±10°) and translations (±3 pixels)
- **Optimizer:** Adam optimizer with learning rate 0.001
- **Training:** 8 epochs

---

## 🏗 CNN Architecture

The network consists of multiple convolutional blocks designed to extract hierarchical spatial features:

- **Input Layer:** $28 \times 28 \times 1$ grayscale images  
- **Convolutional Layers:**  
  - Conv (8 filters, 3×3)  
  - Conv (16 filters, 3×3)  
  - Conv (32 filters, 3×3)  
  Each followed by Batch Normalization and ReLU activation  
- **Pooling:** Max Pooling (stride 2) for downsampling  
- **Regularization:** Dropout (20%)  
- **Output Layer:** Fully Connected layer with Softmax (10 classes)

---

## 📊 Results and Analysis

- **Accuracy:** High classification performance across all digit classes  
- **Best Performing Digits:** 0, 2, 5, and 7 show near-perfect recognition  
- **Confusion Matrix:** Minor misclassification between similar digits (e.g., 3 vs 5, 8 vs 9)  
- **Generalization:** The model remains robust under handwriting variations due to augmentation and dropout  

---

## 🧠 Conclusion

This project demonstrates an effective CNN-based approach for handwritten digit recognition in MATLAB.  
By integrating Batch Normalization, Dropout, and Data Augmentation, the model achieves strong generalization and stable performance.

The architecture provides a solid foundation for more advanced computer vision applications.

---

## 📌 Future Improvements

- Deeper CNN architectures
- Learning rate scheduling
- Real-time digit recognition system
- Deployment as a MATLAB app or web interface
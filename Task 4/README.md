# Task 4: Hand Gesture Recognition 🤚🎯

**Objective**  
Build a deep learning model to recognize and classify 10 distinct hand gestures from image data, enabling intuitive gesture-based controls.

**Tools & Libraries**  
- Python, NumPy  
- TensorFlow & Keras (MobileNetV2 transfer learning)  
- Matplotlib & Seaborn for visualization  
- kagglehub for dataset download  

---

## 🔍 Overview of Steps

1. **Dataset Download & Restructuring**  
   - Download “leapgestrecog” from Kaggle via `kagglehub`  
   - Automatically locate subject folders, then flatten into `gesture_name/image` structure  
   - Use only 1 subject for faster experimentation  

2. **Data Loading & Visualization**  
   - Create `tf.data` pipelines with an 80/20 train-validation split  
   - Display sample images for each of the 10 gesture classes  

3. **Performance Optimization**  
   - Cache, shuffle, and prefetch datasets for smooth training  

4. **Model Building (Transfer Learning)**  
   - Use pretrained MobileNetV2 (frozen) as a feature extractor  
   - Add data augmentation, rescaling, pooling, dropout, and a softmax head  

5. **Training & Fine-Tuning**  
   - **Phase 1**: Train new head for 3 epochs (feature extraction)  
   - **Phase 2**: Unfreeze last layers and fine-tune for 2 additional epochs  

6. **Evaluation & Metrics**  
   - Achieved **99.00%** validation accuracy  
   - Display confusion matrix and per-class precision/recall (all > 0.92)  
   - Classification report confirms strong performance across all 10 gestures  

7. **Inference on New Images**  
   - Utility function `predict_new_image(model, img_path, class_names)`  
   - Shows predicted label with confidence on any input image  

---

## 📈 Results Snapshot

- **Final Validation Accuracy**: 99.00%  
- All classes achieved > 92% recall and precision.  
- Confusion matrix shows near-perfect classification across gestures.

---

## 🔧 Usage

1. **Install dependencies**:  
   ```bash
   pip install tensorflow keras numpy matplotlib seaborn kagglehub
Run the end-to-end script:
python main.py

Predict on your own image:
from main import predict_new_image
predict_new_image(model, '/path/to/your/gesture.png', class_names)

# Task 3: Cats vs. Dogs Image Classification 🐱🦮

**Objective**  
Implement an SVM classifier to distinguish between cat and dog images using a balanced subset from the Kaggle “Dogs vs. Cats” dataset.

**Tools & Libraries**  
- Python, NumPy, Pandas  
- Scikit-Learn (SVC, train_test_split, metrics)  
- PIL for image handling  
- Matplotlib & Seaborn for visualization  
- kagglehub for streamlined dataset download  

---

## 🔍 Overview of Steps

1. **Dataset Download & Setup**  
   - Use `kagglehub.dataset_download("salader/dogs-vs-cats")`  
   - Data is ready in `train/cats` and `train/dogs` folders  

2. **Image Loading & Preprocessing**  
   - Randomly sample 1,000 images per class to keep runtime reasonable  
   - Convert to grayscale, resize to 64×64 pixels  
   - Flatten and normalize pixel values to [0,1]  

3. **Train/Test Split**  
   - 75% training, 25% testing  
   - Stratify to maintain equal cat/dog proportions  

4. **Train SVM Model**  
   - Use an RBF kernel (`kernel='rbf'`) with regularization `C=1.0`  
   - Fit on flattened image vectors  

5. **Evaluation & Metrics**  
   - Compute **Accuracy** on test set  
   - Display **Classification Report** (precision, recall, F1-score)  
   - Plot **Confusion Matrix** with Seaborn heatmap  

6. **Visual Test**  
   - Randomly select a test image  
   - Show true vs. predicted label, colored green if correct, red if not  

---

## 📈 Results

- **Test Accuracy**: 60.00%  
- **Precision / Recall / F1** (both classes): 0.60  
- The confusion matrix shows balanced performance across cats and dogs.

> *Note: SVM with raw pixels can be slow and limited; experiment with feature extraction or CNNs for higher accuracy.*

---

## 🔧 Usage

1. **Install dependencies**:  
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn pillow kagglehub
Run the script:

python main.py

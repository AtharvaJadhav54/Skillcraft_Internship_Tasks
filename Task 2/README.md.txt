# Task 2: Customer Segmentation 🛍️

**Objective**  
Group retail customers into meaningful segments based on their annual income and spending behavior using K-means clustering.

**Tools & Libraries**  
- Python, Pandas, NumPy  
- Scikit-Learn (KMeans, StandardScaler, silhouette_score)  
- Matplotlib & Seaborn for plotting  

---

## 🔍 Overview of Steps

1. **Data Loading & Inspection**  
   - Load `Mall_Customers.csv`  
   - Preview first few rows and verify file presence  
2. **Data Preparation**  
   - Rename columns for clarity (`Annual_Income`, `Spending_Score`)  
   - Select features and standardize them with `StandardScaler`  
3. **Optimal k Determination**  
   - Compute WCSS for k = 2…10 (Elbow Method)  
   - Calculate silhouette scores to gauge cluster quality  
   - Save both plots as `optimal_k_plots.png`  
4. **Apply K-Means & Label Data**  
   - Choose k = 5 based on elbow and silhouette  
   - Fit KMeans, assign cluster labels to each customer  
   - Inverse-scale centroids and overlay on scatter plot  
   - Save visualization as `customer_segments_plot.png`  
5. **Cluster Interpretation**  
   - Compute mean Age, Income, Spending per cluster  
   - Map numeric clusters to descriptive segment names:  
     - High Income, Low Spending  
     - Average  
     - High Income, High Spending (Target)  
     - Low Income, Low Spending  
     - Low Income, High Spending  
   - Export final labeled dataset to `customer_segments_output.csv`  

---

## 📈 Results & Insights

- **Optimal clusters (k)**: 5  
- **Segment Profiles (mean values)**:  
  | Segment                              | Age   | Income (k$) | Spending Score |
  |--------------------------------------|-------|-------------|----------------|
  | High Income, Low Spending            | 42.7  | 55.3        | 49.5           |
  | Average                              | 32.7  | 86.5        | 82.1           |
  | High Income, High Spending (Target)  | 25.3  | 25.7        | 79.4           |
  | Low Income, Low Spending             | 41.1  | 88.2        | 17.1           |
  | Low Income, High Spending            | 45.2  | 26.3        | 20.9           |

*This segmentation helps tailor marketing strategies—e.g., target “High Income, High Spending” customers with premium offers.*

---

## 🔧 Usage

1. Ensure `Mall_Customers.csv` is in this folder.  
2. Install dependencies:  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
Run the script:

bash
Copy
Edit
python main.py
View outputs:

optimal_k_plots.png (Elbow & Silhouette)

customer_segments_plot.png (Cluster visualization)

customer_segments_output.csv (Final labeled data)
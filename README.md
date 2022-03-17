# MLProjects 
Description of the project and goals

## Performance
Types of problems addressed in each project.
Specfic solutions used.
|Project|Best Accuracy (%)|Model|
|---|---|---|
|Forest Cover Type|97.8|K-NN|
|Breast Cancer Detection|97|Neural Net|
|Customer Segmentation|-|K-Means|
|Restaurant Review Classification|73|Gaussian NB|
|Credit Card Fraud Detection|99.95|RandomForest|


## Projects
1. Forest Cover Type
   - Remote Sensing Survey of Forest Areas using satellite
    imagery Data
   - Multi-class classification model predicts forest cover type. 
   - Final accuracy 95%, improvement compared to ~70% in the original study.
   - Dataset: https://archive.ics.uci.edu/ml/datasets/Forest+type+mapping
            (University of California Irvine)

2. Breast Cancer Detection
   - Binary Classifier of cell type as malignant/benign, based on biophysical data
   - Dataset: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
            (University of Wisconsin)

3. Customer Segmentation
   - Customer segmentation performed from usage behavior data of credit card customers 
   - Unsupervised Learning, Clustering in to optimal n_clusters
   - K-Means with Elbow method to determine optimum n_clusters
   - Dataset: https://www.kaggle.com/arjunbhasin2013/ccdata

4. Restaurant Review Classification
   - Sentiment Analysis of restaurant reviews using binary classifier (positive/negative).
   - Feature extraction from Text - CountVectorizer, Bag of Words
   - Dataset: https://www.kaggle.com/akram24/restaurant-reviews


5. Credit Card Fraud Detection
   - Credit card transaction variables are used to identify fradulent/regular transactions
   - Unbalanced Dataset: 0.17% positives(fraudlent)
   - Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud (Université Libre de Bruxelles)


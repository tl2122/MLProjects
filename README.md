# MLProjects 
In this repository I have documented application of machine learning techniques to various publically available data. These are implemented in python using scikit-learn, pandas, and numpy packages.

## Performance
Types of problems addressed in each project.
Specfic solutions used.
|Project|Best Accuracy (%)|Model|
|---|---|---|
|Forest Cover Type|97.8|K-NN|
|Breast Cancer Detection|97|Neural Net|
|Customer Segmentation|-|K-Means|
|Restaurant Review Classification|73|Gaussian NB|
|                                |70| Generated embeddings|
|Credit Card Fraud Detection|99.95|RandomForest|
|Traffic Sign Identification|86.8|Deep NN, Conv-net|
|Character-level Language Model|-|RNN, NLP| 

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
   - Also created and tested custom embeddings (tf.keras.layers.Embedding); did not improve on CountVectorizer
   - Dataset: https://www.kaggle.com/akram24/restaurant-reviews


5. Credit Card Fraud Detection
   - Credit card transaction variables are used to identify fradulent/regular transactions
   - Unbalanced Dataset: 0.17% positives(fraudlent)
   - Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud (Université Libre de Bruxelles)

6. Traffic Signs Identification
   - Dataset:  https://www.kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed 
      (German Traffic Sign Recognition Benchmarks (GTSRB))
   - Deep Conv-net with 2 layers and Funtional API in tensorflow/keras.
   - Unnormalized, and color images used here. 85% accuracy achieved. 

7. Character Level Language Model 
   - Built Language model using RNN, on dataset of dinosaur names
   - Tested language model by varying hyperparameters and generating new names.


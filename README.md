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
|                                |70|Generated embeddings|
|Credit Card Fraud Detection|99.95|RandomForest|
|Traffic Sign Identification|86.8|Deep NN, Conv-net|
|Character-level Language Model|-|RNN, NLP| 
|MBTI Classification with Text |77|Bi-Directional RNN,Vectorization,Embeddings, GRU, NLP| 
|Protein Structure|-|RandomForest, MultiOutputClassifier, LSTM|
|Book Recommendation|-|Embeddings, NN|
|Disease Prediction|100|Gaussian NB, Random Forest, SVC|
|Arrhythmia|70|Random Forest, GradientBoosting,SVC, MLP|

## Projects
1. Forest Cover Type
   - Remote Sensing Survey of Forest Areas using satellite
    imagery Data
   - Multi-class classification model predicts forest cover type. 
   - Final accuracy 95%, improvement compared to ~70% in the original study.
   - Dataset: archive.ics.uci.edu/ml/datasets/Forest+type+mapping
            (University of California Irvine)

2. Breast Cancer Detection
   - Binary Classifier of cell type as malignant/benign, based on biophysical data
   - Dataset: kaggle.com/uciml/breast-cancer-wisconsin-data
            (University of Wisconsin)

3. Customer Segmentation
   - Customer segmentation performed from usage behavior data of credit card customers 
   - Unsupervised Learning, Clustering in to optimal n_clusters
   - K-Means with Elbow method to determine optimum n_clusters
   - Dataset: kaggle.com/arjunbhasin2013/ccdata

4. Restaurant Review Classification
   - Sentiment Analysis of restaurant reviews using binary classifier (positive/negative).
   - Feature extraction from Text - CountVectorizer, Bag of Words 
   - Also created and tested custom embeddings (tf.keras.layers.Embedding); did not improve on CountVectorizer
   - Dataset: kaggle.com/akram24/restaurant-reviews


5. Credit Card Fraud Detection
   - Credit card transaction variables are used to identify fradulent/regular transactions
   - Unbalanced Dataset: 0.17% positives(fraudlent)
   - Dataset: kaggle.com/mlg-ulb/creditcardfraud (Université Libre de Bruxelles)

6. Traffic Signs Identification
   - Dataset:  kaggle.com/datasets/valentynsichkar/traffic-signs-preprocessed 
      (German Traffic Sign Recognition Benchmarks (GTSRB))
   - Deep Conv-net with 2 layers and Funtional API in tensorflow/keras.
   - Unnormalized, and color images used here. 85% accuracy achieved. 

7. Character Level Language Model 
   - Built Language model using RNN, on dataset of dinosaur names
   - Tested language model by varying hyperparameters and generating new names.

8. MBTI Classification with Text  
   - Analysis on language styles and online behavior to predict personality type
   - Built and Trained two classifiers: multi-class with full range of classes (INFJ - ESTP), as 
      well as binary introvert vs extrovert classifier.  
   - Vectorization/Tokenization, Embedding, bi-directional RNN with GRU
   - Dataset: kaggle.com/datasets/datasnaek/mbti-type

9. Protein Structure Prediction
   - Protein shape prediction based on amino-acid sequence
   - Multi-class, multi-output, sequence prediction
   - Data:https://archive.ics.uci.edu/ml/datasets

10. Protein Structure Prediction with LSTM
   - Same input as 9. LSTM used for protein sequence to shape modelling

11. Book Recommendation System based on user ratings data.
12. Disease Prediction
13. Arrhythmia detection and classification

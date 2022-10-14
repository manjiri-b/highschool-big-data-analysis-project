# High School Big Data Analysis Project

Dimensions were reduced using PCA from sklearn. It was conducted on those variables that had a moderate to strong correlation, and hence could be classified as general factor(s). Before every PCA, the data were standardized using z-scores, the eigenvalues and loadings of the principal complements were found and graphed. pca.fit_transform(zscoredData) was used to generate rotated data. Rotated data represents each principal component depending on its weight for each individual variable. Data cleaning was done depending on whether it was needed for the question, i.e, if the columns of interest had missing values. Whenever I encountered missing data, I dropped it using .dropna(). For the multiple linear regression model in question 8, the training and test datasets were scaled using MinMaxScaler from scikit.

![alt text](https://user-images.githubusercontent.com/83567562/195762423-23596fe3-f5ba-4a90-b17c-e25387f7beb0.png)

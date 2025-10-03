# AWS Academy Machine Learning Foundations Lab Work

This repository contains the lab work for the **AWS Academy Machine Learning Foundations** course (Course ID: 127335). The labs cover key concepts and hands-on exercises to build foundational knowledge in machine learning and AWS services. The course provided practical experience with using AWS tools for machine learning tasks.


---

## 📝 Table of Content

| File Name                                        | Description                                                                    |
|--------------------------------------------------|--------------------------------------------------------------------------------|
| 3.1 Creating and importing data.ipynb            | Lab 3.1 Creating and importing data                                            |
| 3.2 Exploring Data.ipynb                         | Lab 3.2 Amazon SageMaker - Exploring Data                                      |
| 3.3 Encoding Categorical Data.ipynb              | Lab 3.3 Amazon SageMaker - Encoding Categorical Data                           |
| 3.4 Training a model.ipynb                       | Lab 3.4 - Training a model                                                     |
| 3.5 Deploying a model.ipynb                      | Lab 3.5 - Amazon SageMaker - Deploying a model                                 |
| 3.6 Generating model performance metrics.ipynb   | Lab 3.6 - Amazon SageMaker - Generating model performance metrics              |
| 3.7 Hyperparameter Tuning.ipynb                  | Lab 3.7 - Amazon SageMaker - Hyperparameter Tunings                            |
| Challenge Lab Flight_Delay-Student.ipynb         | Challenge Lab Predicting Airplane Delays                                       |


---

## Course Overview

The **AWS Academy Machine Learning Foundations** course introduces students to the concepts and terminology of Artificial Intelligence and machine learning. By the end of this course, students will be able to select and apply machine learning services to resolve business problems. They will also be able to label, build, train, and deploy a custom machine learning model through a guided, hands-on approach.


## Lab Instructions

### Lab 3.1: **Creating and Importing Data**
**Objective:**  
Learn how to create and import datasets into Amazon SageMaker for machine learning tasks.

**Instructions:**  
1. Open the notebook `3.1 Creating and importing data.ipynb`.
2. Load a sample dataset into the environment using AWS S3.
3. Explore methods to read CSV files into a Pandas DataFrame and inspect the contents.
4. Perform basic data validation and ensure that the data is clean and ready for preprocessing.
5. Save the cleaned data into an S3 bucket for further use in subsequent labs.

---

### Lab 3.2: **Exploring Data**
**Objective:**  
Learn how to explore data using basic descriptive statistics and visualizations.

**Instructions:**  
1. Open the notebook `3.2 Exploring Data.ipynb`.
2. Load the cleaned dataset from S3 that was created in Lab 3.1.
3. Use `pandas` and `matplotlib` to explore basic statistical metrics (mean, median, standard deviation, etc.) for each feature in the dataset.
4. Generate histograms and box plots to visualize data distributions.
5. Check for missing values and handle them by either imputation or removal.
6. Save any visualizations you generate and add them to the notebook for reference.

---

### Lab 3.3: **Encoding Categorical Data**
**Objective:**  
Learn how to encode categorical variables into a numerical format so they can be used in machine learning models.

**Instructions:**  
1. Open the notebook `3.3 Encoding Categorical Data.ipynb`.
2. Load the dataset created in the previous labs.
3. Identify categorical columns that need encoding (e.g., strings like 'Male' or 'Female').
4. Use `pandas` to apply one-hot encoding and label encoding to these categorical features.
5. Verify the encoded data by checking the first few rows.
6. Save the encoded data for future use.

---

### Lab 3.4: **Training a Model**
**Objective:**  
Train a machine learning model on the processed data using Amazon SageMaker.

**Instructions:**  
1. Open the notebook `3.4 Training a model.ipynb`.
2. Split the dataset into training and testing sets.
3. Choose a machine learning algorithm (e.g., logistic regression, decision tree, or random forest).
4. Train the model using the training dataset.
5. Evaluate the model's performance on the test dataset.
6. Use performance metrics such as accuracy, precision, recall, and F1-score to evaluate the model.

---

### Lab 3.5: **Deploying a Model**
**Objective:**  
Deploy your trained model to Amazon SageMaker for real-time inference.

**Instructions:**  
1. Open the notebook `3.5 Deploying a model.ipynb`.
2. Save your trained model to Amazon S3.
3. Create a SageMaker endpoint for real-time inference.
4. Deploy the trained model to the endpoint.
5. Test the model by sending sample requests to the deployed endpoint.

---

### Lab 3.6: **Generating Model Performance Metrics**
**Objective:**  
Generate additional performance metrics for your trained model to better understand its performance.

**Instructions:**  
1. Open the notebook `3.6 Generating model performance metrics.ipynb`.
2. Use the test dataset to generate additional performance metrics (e.g., confusion matrix, ROC curve, AUC score).
3. Visualize the ROC curve and compute the area under the curve (AUC).
4. Save the visualizations and metrics for reference.

---

### Lab 3.7: **Hyperparameter Tuning**
**Objective:**  
Learn how to tune hyperparameters of a machine learning model using SageMaker's hyperparameter tuning functionality.

**Instructions:**  
1. Open the notebook `3.7 Hyperparameter Tuning.ipynb`.
2. Define a set of hyperparameters to tune (e.g., learning rate, number of estimators).
3. Set up a SageMaker Hyperparameter Tuning Job to search for the best hyperparameter combination.
4. Evaluate the results of the tuning job and compare performance with the baseline model.
5. Save the best performing model for deployment.

---

### Challenge Lab: **Predicting Airplane Delays**
**Objective:**  
Build a predictive model to forecast airplane delays based on real-world data.

**Instructions:**  
1. Open the notebook `Challenge Lab Flight_Delay-Student.ipynb`.
2. Load the flight delay dataset.
3. Perform exploratory data analysis (EDA) to understand the dataset.
4. Preprocess the data by handling missing values, encoding categorical variables, and scaling numerical features.
5. Split the dataset into training and testing sets.
6. Train a machine learning model (e.g., logistic regression, gradient boosting) to predict flight delays.
7. Evaluate the model’s performance using accuracy, precision, recall, and F1-score.

---

## Technologies Used

- **AWS SageMaker**: For building, training, and deploying machine learning models.
- **Python**: Programming language used in the labs for data manipulation and model training.
- **Jupyter Notebooks**: Used to document and run the lab exercises.
- **Pandas**: For data manipulation and cleaning.
- **NumPy**: For numerical computations.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-Learn**: For machine learning algorithms and utilities.


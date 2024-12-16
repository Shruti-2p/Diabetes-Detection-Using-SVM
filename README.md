# Diabetes Detection Using SVM

This project involves building a machine learning model to detect diabetes in patients using the Pima Indians Diabetes dataset. The core algorithm used is Support Vector Machines (SVM), which is optimized through hyperparameter tuning to ensure high accuracy and reliability.

### Overview of the Project:
The primary goal is to create a system that can classify individuals as diabetic or non-diabetic based on various medical measurements such as glucose levels, insulin levels, and BMI, among others. The project also includes a web-based application built with Streamlit, allowing users to input data and receive predictions instantly.

### Key Features:
1. **Dataset Handling**:
   - The Pima Indians Diabetes dataset includes 8 key attributes like pregnancies, plasma glucose, BMI, and age.
   - The target variable indicates diabetes presence (1) or absence (0).

2. **Machine Learning Model**:
   - Implements SVM for classification.
   - Hyperparameters like kernel type, regularization parameter (C), and gamma are fine-tuned using GridSearchCV.

3. **Evaluation Metrics**:
   - Accuracy score to measure model performance.
   - Confusion matrix to visualize predictions versus actual outcomes.

4. **Web Application**:
   - A user interface built with Streamlit to input features and display predictions.
   - Allows non-technical users to interact with the model seamlessly.

5. **Model Serialization**:
   - The trained SVM model is saved using Python’s pickle module, making it easy to deploy and reuse.

### Technologies Used:
- **Programming Language**: Python
- **Libraries**: pandas, NumPy, scikit-learn, matplotlib, Streamlit
- **Tools**: GridSearchCV for hyperparameter tuning, pickle for model serialization

### Achievements:
- The SVM model achieves an accuracy of **93%** on the test set.
- Optimization through hyperparameter tuning further improves the model's reliability and performance.

### Usage:
1. Train the model using the provided script to generate predictions.
2. Launch the Streamlit web application to make predictions interactively.

### Future Scope:
- Experiment with other machine learning algorithms for enhanced performance.
- Incorporate additional data features to improve the prediction capability.
- Deploy the web application on a cloud platform for wider accessibility.


# Employee Attrition Prediction - Machine Learning Web Application

## Overview

This project aims to build a machine learning model that predicts employee attrition (whether an employee will leave the company) based on various features. The model is then integrated into a user-friendly web application using Flask, allowing users to input employee information and receive predictions instantly. The web application also includes a feature to upload a CSV file containing new employee details for batch predictions.

## Steps and Components

1. **Data Preparation and Model Building:**
   - **Data Collection:** Gathered employee data from the company's records.
   - **Data Preprocessing:** Cleaned the data by handling missing values, transforming categorical variables, and removing irrelevant features.
   - **Feature Engineering:** Selected relevant features and encoded categorical variables using one-hot encoding.
   - **Train-Test Split:** Divided the data into training and testing sets for model evaluation.
   - **Handling Imbalanced Data:** Used SMOTE technique to balance the target classes.
   - **Scaling:** Scaled numerical features using StandardScaler.
   - **Model Selection and Training:** Chose a Support Vector Machine (SVM) classifier with a linear kernel and trained it on the training data.
   - **Model Evaluation:** Evaluated the model's performance using accuracy, precision, recall, and confusion matrix.

2. **Web Application Development:**
   - **Flask Setup:** Created a Flask app with routes for different functionalities.
   - **HTML Templates:** Designed HTML templates for the web application's input and output pages.
   - **CSS Styling:** Styled the web pages using CSS for an appealing and user-friendly interface.
   - **Form Handling:** Implemented form submission and handling routes to process user input.
   - **File Upload Feature:** Developed a feature to upload CSV files containing new employee data for batch predictions.

3. **Model Integration and Deployment:**
   - **Load and Train Model:** Implemented a function to load the trained SVM model and preprocessing steps.
   - **Process Form Data:** Processed user input, preprocessed it, and used the model for predictions.
   - **Display Predictions:** Displayed predictions on the webpage for immediate feedback.
   - **Data Upload Feature:** Allowed users to upload CSV files with new employee data, processed the data, and provided predictions for the uploaded records.

4. **Local Testing and Deployment:**
   - **Local Testing:** Tested the web application locally using VSCode to ensure all functionalities work as intended.
   - **Deployment:** Deployed the Flask web application on a cloud platform (e.g., Heroku, AWS, Azure).
   - **Sharing with Team:** Shared the deployed web application's URL with the team for testing and feedback.

## Future Enhancements

- **Model Improvement:** Explore different machine learning algorithms and hyperparameter tuning to improve model performance.
- **UI Enhancement:** Further enhance the user interface with more user-friendly features and visualizations.
- **Predictive Insights:** Provide additional insights or recommendations based on the model's predictions.
- **User Authentication:** Implement user authentication to restrict access to authorized personnel.

By combining machine learning techniques and web development skills, this project offers a practical solution for predicting employee attrition and making data-driven decisions within the organization. The web application simplifies the process of receiving predictions and can be a valuable tool for HR departments and management teams.

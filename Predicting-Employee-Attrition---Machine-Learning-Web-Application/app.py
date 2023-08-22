from flask import Flask, render_template, request
from pandas import read_csv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
import pandas as pd

app = Flask(__name__)

# Create a global variable for the fitted StandardScaler
scaler = StandardScaler()

def load_and_train_model():
    data = read_csv(r'H:\MSC in BA\Applied Statistics & Machine Learning\Project\Employee Prediction\Employee Prediction_2\Predicting-Employee-Attrition---Machine-Learning-Web-Application\Sample_Employee_Data.csv')  # Replace with your CSV file path

    # Preprocess categorical variables using Label Encoding
    label_encoders = {}
    categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le

    data['PastEmployee'] = data['PastEmployee'].map({'Yes': 1, 'No': 0})
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
    data['Gender'] = data['Gender'].map({'Female': 1, 'Male': 0})

    X = data.drop('PastEmployee', axis=1)
    Y = data['PastEmployee']
    X_scaled = scaler.fit_transform(X)  # Fit the scaler here
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.25, random_state=100)
    X_train, Y_train = SMOTE(random_state=100).fit_resample(X_train, Y_train)

    svm_classifier = SVC(kernel='linear', random_state=0)
    svm_classifier.fit(X_train, Y_train)

    return svm_classifier, label_encoders

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load model and preprocessors
    trained_model, label_encoders = load_and_train_model()

    # Process the form data and make a prediction
    age = float(request.form['age'])
    past_employee = request.form['past_employee']
    business_travel = request.form['business_travel']
    department = request.form['department']
    distance_from_home = float(request.form['distance_from_home'])
    education_field = request.form['education_field']
    environment_satisfaction = int(request.form['environment_satisfaction'])
    gender = request.form['gender']
    job_role = request.form['job_role']
    marital_status = request.form['marital_status']
    monthly_income = float(request.form['monthly_income'])
    num_companies_worked = float(request.form['num_companies_worked'])
    over_time = request.form['over_time']

    # Encode categorical values using Label Encoders
    business_travel_encoded = label_encoders['BusinessTravel'].transform([business_travel])[0]
    department_encoded = label_encoders['Department'].transform([department])[0]
    education_field_encoded = label_encoders['EducationField'].transform([education_field])[0]
    job_role_encoded = label_encoders['JobRole'].transform([job_role])[0]
    marital_status_encoded = label_encoders['MaritalStatus'].transform([marital_status])[0]
    gender_encoded = 1 if gender == 'Female' else 0
    over_time_encoded = 1 if over_time == 'Yes' else 0

    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'Age': [age],
        'BusinessTravel': [business_travel_encoded],
        'Department': [department_encoded],
        'DistanceFromHome': [distance_from_home],
        'EducationField': [education_field_encoded],
        'EnvironmentSatisfaction': [environment_satisfaction],
        'Gender': [gender_encoded],
        'JobRole': [job_role_encoded],
        'MaritalStatus': [marital_status_encoded],
        'MonthlyIncome': [monthly_income],
        'NumCompaniesWorked': [num_companies_worked],
        'OverTime': [over_time_encoded]
    })

    # Scale input data using the fitted StandardScaler
    preprocessed_data = scaler.transform(input_data)

    # Make a prediction
    prediction = trained_model.predict(preprocessed_data)

    return render_template('index.html', prediction=prediction)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        data_file = request.files['data_file']
        if data_file:
            try:
                data = pd.read_csv(data_file)
                # You can now process the uploaded data using your model
                # Example: prediction = trained_model.predict(data)
                # Replace 'trained_model' with your actual trained model
                # You can return the prediction result or any other output

                # For demonstration purposes, let's return the uploaded data
                return render_template('index.html', uploaded_data=data.to_html(index=False))
            except Exception as e:
                return "Error processing the data file."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

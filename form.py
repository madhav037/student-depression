import streamlit as st
import pickle
import numpy as np

# Load your model and encoders
def load_model_and_encoders():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open('labelEncoder.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    
    return model, encoders

# Function to encode user input using loaded encoders
def encode_input(user_inputs, encoders):
    encoded_inputs = []
    for col, value in user_inputs.items():
        if col in encoders:
            # Check if it's a categorical feature and use LabelEncoder
            value = encoders[col].transform([value])[0]  # Encode categorical input
        encoded_inputs.append(value)
    return encoded_inputs

# Predict function
def predict(model, user_inputs, encoders):
    # Encode the input data
    encoded_inputs = encode_input(user_inputs, encoders)
    
    # Convert input data to numpy array and make the prediction
    encoded_inputs = np.array(encoded_inputs).reshape(1, -1)  # Reshape for a single prediction
    print(encoded_inputs)
    prediction = model.predict(encoded_inputs)
    
    return prediction

# Streamlit form for user input
st.title("Mental Health Data Form")

with st.form("mental_health_form"):
    st.header("Personal Details")
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", min_value=18, max_value=59, step=1)
    city = st.selectbox("City", [
        'Visakhapatnam', 'Bangalore', 'Srinagar', 'Varanasi', 'Jaipur', 'Pune',
        'Thane', 'Chennai', 'Nagpur', 'Nashik', 'Vadodara', 'Kalyan', 'Rajkot',
        'Ahmedabad', 'Kolkata', 'Mumbai', 'Lucknow', 'Indore', 'Surat', 
        'Ludhiana', 'Bhopal', 'Meerut', 'Agra', 'Ghaziabad', 'Hyderabad',
        'Vasai-Virar', 'Kanpur', 'Patna', 'Faridabad', 'Delhi'
    ])
    profession = st.selectbox("Profession", [
        'Student', 'Civil Engineer', 'Architect', 'UX/UI Designer',
        'Digital Marketer', 'Content Writer', 'Educational Consultant',
        'Teacher', 'Manager', 'Chef', 'Doctor', 'Lawyer', 'Entrepreneur', 
        'Pharmacist'
    ])
    degree = st.selectbox("Degree", [
        'B.Pharm', 'BSc', 'BA', 'BCA', 'M.Tech', 'PhD', 'Class 12', 'B.Ed', 
        'LLB', 'BE', 'M.Ed', 'MSc', 'BHM', 'M.Pharm', 'MCA', 'MA', 'B.Com', 
        'MD', 'MBA', 'MBBS', 'M.Com', 'B.Arch', 'LLM', 'B.Tech', 'BBA', 'ME', 
        'MHM', 'Others'
    ])
    
    st.header("Health & Lifestyle")
    
    academic_pressure = st.slider("Academic Pressure (0-5)", min_value=0, max_value=5, step=1)
    work_pressure = st.slider("Work Pressure (0-5)", min_value=0, max_value=5, step=1)
    cgpa = st.slider("CGPA (0-10)", min_value=0.0, max_value=10.0, step=0.1)
    study_satisfaction = st.slider("Study Satisfaction (0-5)", min_value=0, max_value=5, step=1)
    job_satisfaction = st.slider("Job Satisfaction (0-4)", min_value=0, max_value=4, step=1)
    sleep_duration = st.selectbox("Sleep Duration", [
        "5-6 hours", "Less than 5 hours", "7-8 hours", "More than 8 hours", "Others"
    ])
    dietary_habits = st.selectbox("Dietary Habits", [
        "Healthy", "Moderate", "Unhealthy", "Others"
    ])
    work_study_hours = st.slider("Work/Study Hours (0-12)", min_value=0, max_value=12, step=1)
    financial_stress = st.slider("Financial Stress (1-5)", min_value=1, max_value=5, step=1)
    
    st.header("Mental Health")
    
    suicidal_thoughts = st.radio(
        'Have you ever had suicidal thoughts ?', ["Yes", "No"]
    )
    family_history = st.radio(
        'Family History of Mental Illness', ["Yes", "No"]
    )
    # Submit button
    submit_button = st.form_submit_button("Submit")

# Load model and encoders
model, encoders = load_model_and_encoders()

# Display results if the form is submitted
if submit_button:
    st.success("Form submitted successfully!")
    input_data = {
        "Gender": gender,
        "Age": age,
        "City": city,
        "Profession": profession,
        "Academic Pressure": academic_pressure,
        "Work Pressure": work_pressure,
        "CGPA": cgpa,
        "Study Satisfaction": study_satisfaction,
        "Job Satisfaction": job_satisfaction,
        "Sleep Duration": sleep_duration,
        "Dietary Habits": dietary_habits,
        "Degree": degree,
        "Have you ever had suicidal thoughts ?": suicidal_thoughts,
        "Work/Study Hours": work_study_hours,
        "Financial Stress": financial_stress,
        "Family History of Mental Illness": family_history,
    }

    # Make prediction
    prediction = predict(model, input_data, encoders)

    st.write("Predicted Mental Health Risk: ","Depressed" if prediction == 1 else 'Good no depression yet!')

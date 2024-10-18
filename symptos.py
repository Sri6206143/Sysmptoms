import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.svm import SVC

# Load the dataset
dataset = pd.read_csv('dataset/Training.csv')

# Separate features and target
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']

# Encode the target (prognosis)
le = LabelEncoder()
Y = le.fit_transform(y)

# Save the LabelEncoder
pickle.dump(le, open('label_encoder.pkl', 'wb'))  # Save the encoder

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Train the SVC model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Save the trained model using pickle
pickle.dump(svc, open('svc.pkl', 'wb'))

# Load the saved SVC model
svc = pickle.load(open('svc.pkl', 'rb'))

# Load the supplementary data files
sym_des = pd.read_csv("dataset/symtoms_df.csv")
precautions = pd.read_csv("dataset/precautions_df.csv")
workout = pd.read_csv("dataset/workout_df.csv")
description = pd.read_csv("dataset/description.csv")
medications = pd.read_csv('dataset/medications.csv')
diets = pd.read_csv("dataset/diets.csv")

# Helper function to get disease-related recommendations
def helper(dis):
    desc = description[description['Disease'] == dis]
    if not desc.empty:
        desc = desc['Description'].values[0]
    else:
        desc = "Description not available."
    
    precautions_filtered = precautions[precautions['Disease'] == dis]
    if not precautions_filtered.empty:
        pre = precautions_filtered[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values.flatten()
    else:
        pre = ["No precautions available."] * 4  # Default message for precautions
    
    med = medications[medications['Disease'] == dis]
    if not med.empty:
        medication = med['Medication'].values[0]
    else:
        medication = "Medication not available."
        
    die = diets[diets['Disease'] == dis]
    if not die.empty:
        diet = die['Diet'].values[0]
    else:
        diet = "Diet information not available."
        
    wrkout = workout[workout['disease'] == dis]
    if not wrkout.empty:
        workout_plan = wrkout['workout'].values[0]
    else:
        workout_plan = "Workout plan not available."
    
    return desc, pre, medication, diet, workout_plan

# Streamlit UI
st.title("AI Healthcare Assistant")
st.write("Enter your symptoms below:")

# Text input for symptoms
user_input = st.text_area("Symptoms (comma-separated):")

# Button to predict disease
if st.button("Predict Disease"):
    # Preprocess user input
    symptom_list = user_input.split(',')
    symptom_list = [symptom.strip() for symptom in symptom_list]
    
    # Create a feature vector based on user input
    symptom_vector = [1 if symptom in symptom_list else 0 for symptom in dataset.columns[:-1]]  # assuming last column is prognosis
    symptom_vector = pd.DataFrame([symptom_vector])

    # Make a prediction
    predicted_disease = svc.predict(symptom_vector)[0]
    
    # Decode the predicted disease
    predicted_disease_decoded = le.inverse_transform([predicted_disease])[0]

    # Get recommendations for the predicted disease
    desc, precautions, medication, diet, workout_plan = helper(predicted_disease_decoded)

    # Display results
    st.write(f"**Predicted Disease:** {predicted_disease_decoded}")
    st.write(f"**Description:** {desc}")
    st.write(f"**Precautions:** {', '.join(precautions)}")
    st.write(f"**Medication:** {medication}")
    st.write(f"**Diet:** {diet}")
    st.write(f"**Workout Plan:** {workout_plan}")

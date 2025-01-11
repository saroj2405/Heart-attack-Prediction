import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


df = pd.read_csv("heart_attack_prediction_dataset.csv")


st.title("HeartWise: Risk Prediction🫀⚠️")
st.sidebar.title("Content")

options = st.sidebar.selectbox(
    'Choose a section:',
    ('Dataset Overview', 'Exploratory Data Analysis', 'Make Predictions')
)


if options == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("**First 5 rows of the dataset:**")
    st.dataframe(df.head())  # Display the first 5 rows of the data
    st.write("**Summary Statistics:**")
    st.dataframe(df.describe())
    st.write(f"**Total Items:** {df.shape[0]}")
    st.write("**Data Types of Each Column:**")
    st.dataframe(df.dtypes)

elif options == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    analysis_type = st.sidebar.selectbox('Select Unvivarate/Bivarate Analysis',('Univarate', 'Bivarate'))
    
    if analysis_type == 'Univarate':
        st.subheader("Univarate Analysis - Numerical")
        col_name = st.selectbox('Select Any Numerical Column',('Age', 'BMI', 'Cholesterol', 'Exercise Hours Per Week', 'Heart Rate', 'Physical Activity Days Per Week',  'Sedentary Hours Per Day', 'Sleep Hours Per Day', 'Stress Level', 'Triglycerides'))
        st.bar_chart(df[col_name].value_counts())
        st.header("Univarate Analysis - Categorical")

        col1, col2 = st.columns(2)
        with col1:
            col_value_count = df['Sex'].value_counts()
            sizes = list(col_value_count.values)
            labels = col_value_count.index.to_list()  
            colors = sns.color_palette("Set2", len(labels))  
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90,colors=colors)
            ax1.axis('equal')
            ax1.set_title('Gender Distribution')
            st.pyplot(fig1)
        with col2:
            col_value_count = df['Smoking'].value_counts()
            sizes = list(col_value_count.values)
            labels = col_value_count.index.to_list() 
            colors = sns.color_palette("Pastel1", len(labels))   
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes,labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90,colors=colors)
            ax1.axis('equal')
            ax1.set_title('Smoking Status Distribution')
            st.pyplot(fig1)

        col1, col2 = st.columns(2)
        with col1:
            col_value_count = df['Diabetes'].value_counts()
            sizes = list(col_value_count.values)
            labels = col_value_count.index.to_list() 
            colors = sns.color_palette("coolwarm", len(labels))    
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90,colors=colors)
            ax1.axis('equal')
            ax1.set_title('Diabetes Distribution')
            st.pyplot(fig1)
        with col2:
            col_value_count = df['Alcohol Consumption'].value_counts()
            sizes = list(col_value_count.values)
            labels = col_value_count.index.to_list()   
            colors = sns.color_palette("Paired", len(labels))  
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=90,colors=colors)
            ax1.axis('equal')
            ax1.set_title('Alcohol Consumption Distribution')
            st.pyplot(fig1)
    
    elif analysis_type=='Bivarate':
        st.subheader("Bivarate Analysis Against `Heart Attack Risk`")
        col =  st.selectbox('Select One from below:',('Sex','Smoking','Diabetes','Alcohol Consumption','Medication Use', 'Obesity','Family History'))
        fig, ax = plt.subplots(figsize=(10, 5))
        # Using seaborn to create a grouped bar plot
        sns.countplot(x=col, hue='Heart Attack Risk', data=df, ax=ax, palette='Set2')
        ax.set_xlabel('Smoking Status')
        ax.set_ylabel('Count')
        ax.set_title('Smoking vs Heart Attack Risk')
        st.pyplot(fig)
    
elif options == "Make Predictions":
    with open('model.pkl', 'rb') as f:
        loaded_reg = pickle.load(f)

    col1, col2 = st.columns(2)
    with col1:
        cholesterol = st.number_input('Cholesterol', min_value=100, max_value=300, value=217)
        blood_pressure = st.number_input('Blood Pressure', min_value=60, max_value=180, value=66)
    with col2:
        heart_rate = st.number_input('Heart Rate', min_value=40, max_value=200, value=75)
        bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=35.18)

    col1, col2 = st.columns(2)

    with col1:
        exercise_hours_per_week = st.slider('Exercise Hours Per Week', min_value=0.0, max_value=50.0, value=17.1)
        stress_level = st.slider('Stress Level', min_value=1, max_value=10, value=10)
        sedentary_hours_per_day = st.slider('Sedentary Hours Per Day', min_value=0.0, max_value=24.0, value=1.73)
    with col2:    
        triglycerides = st.slider('Triglycerides', min_value=50, max_value=500, value=544)
        physical_activity_days_per_week = st.slider('Physical Activity Days Per Week', min_value=0, max_value=7, value=3)
        sleep_hours_per_day = st.slider('Sleep Hours Per Day', min_value=3, max_value=12, value=6)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        diabetes = st.radio('Diabetes', ['No', 'Yes'], index=1)
    with col2:
        family_history = st.radio('Family History', ['No', 'Yes'], index=1)
    with col3:
        smoking = st.radio('Smoking', ['No', 'Yes'], index=1)
    with col4:
        obesity = st.radio('Obesity', ['No', 'Yes'], index=1)
    with col5:
        alcohol_consumption = st.radio('Alcohol', ['No', 'Yes'], index=1)
    with col6:
        medication_use = st.radio('Medication Use', ['No', 'Yes'], index=1)

    previous_heart_problems = st.selectbox('Previous Heart Problems', ('No', 'Yes'))
        
    predict_button_css = """
        <style>
            .stButton > button {
                background-color: #f5370e; /* Green background similar to radio button color */
                color: white;
                width: 100%;
                height: 50px; /* Match the height of the radio button */
                border-radius: 5px;
                font-size: 16px;
                border: none;
            }
            .stButton > button:hover {
                background-color: #f5370e; /* Darker green on hover */

        </style>
    """

    # Inject the custom CSS into the Streamlit app
    st.markdown(predict_button_css, unsafe_allow_html=True)

    

    input_data = {
        'Cholesterol': cholesterol,
        'Blood Pressure': blood_pressure,
        'Heart Rate': heart_rate,
        'Diabetes': 1 if diabetes == 'Yes' else 0,
        'Family History': 1 if family_history == 'Yes' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'Obesity': 1 if obesity == 'Yes' else 0,
        'Alcohol Consumption': 1 if alcohol_consumption == 'Yes' else 0,
        'Exercise Hours Per Week': exercise_hours_per_week,
        'Previous Heart Problems': 1 if previous_heart_problems == 'Yes' else 0,
        'Medication Use': 1 if medication_use == 'Yes' else 0,
        'Stress Level': stress_level,
        'Sedentary Hours Per Day': sedentary_hours_per_day,
        'BMI': bmi,
        'Triglycerides': triglycerides,
        'Physical Activity Days Per Week': physical_activity_days_per_week,
        'Sleep Hours Per Day': sleep_hours_per_day
    }

    # If the model is loaded and ready, make predictions
    if st.button('Predict'):
        # Convert input_data to pandas DataFrame for prediction
        import pandas as pd
        input_df = pd.DataFrame([input_data])

        # Assuming the model is preloaded as 'loaded_reg'
        prediction = loaded_reg.predict(input_df)
        
        prediction_result = "Heart Disease Likely" if prediction[0] == 1 else "No Heart Disease Likely"
        bgcolor = "rgba(255, 99, 71, 0.8)" if prediction[0] == 1 else "rgba(0, 128, 0, 0.8)"
        from streamlit_card import card
        res = card(
            title=prediction_result,
            text= '',
            styles={
                "card": {
                    "width": "100%",
                    "height": "100px"
                },
                "filter": {
                    "background-color": bgcolor
                }
            }
        )

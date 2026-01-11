import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import datetime
import lightgbm as lgb


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Bike Demand Prediction",
    layout="centered"
)

# ---------- LOAD MODEL ----------
model = pickle.load(open("Lgbmodel.pkl", "rb"))

# ---------- UI ----------
st.title("🚲 Bike Demand Prediction App")
st.write("Predict bike demand by entering feature values or uploading a CSV file.")

# ---------- INPUT METHOD ----------
input_type = st.radio(
    "Select Input Method",
    ["Manual Input", "Upload CSV / Excel File"]
)

# ==========================================
# ========== MANUAL INPUT ===================
# ==========================================
if input_type == "Manual Input":

    st.subheader("Input Details")

    def user_input_parameters():
        date = st.sidebar.date_input("Date", datetime.date.today())
        holiday = st.sidebar.selectbox("Is it a Holiday?", [0, 1])
        workingday = st.sidebar.selectbox("Is it a Working Day?", [0, 1])

        weather = st.sidebar.selectbox(
            "Weather Condition",
            ["Clear", "Mist", "Light Snow", "Heavy Rain"]
        )

        season = st.sidebar.selectbox(
            "Season (1=Spring, 2=Summer, 3=Fall, 4=Winter)",
            [1, 2, 3, 4]
        )

        hr = st.sidebar.number_input("Hour", 0, 23)
        weekday = st.sidebar.number_input("Weekday (0=Sun, 6=Sat)", 0, 6)
        temp = st.sidebar.number_input("Temperature")
        atemp = st.sidebar.number_input("Feels-like Temperature")
        hum = st.sidebar.number_input("Humidity (0–1)", 0.0, 1.0)
        windspeed = st.sidebar.number_input("Windspeed")

        data = {
            "holiday": holiday,
            "workingday": workingday,
            "weathersit_Clear": 1 if weather == "Clear" else 0,
            "weathersit_Mist": 1 if weather == "Mist" else 0,
            "weathersit_Light Snow": 1 if weather == "Light Snow" else 0,
            "weathersit_Heavy Rain": 1 if weather == "Heavy Rain" else 0,
            "season": season,
            "hr": hr,
            "weekday": weekday,
            "temp": temp,
            "atemp": atemp,
            "hum": hum,
            "windspeed": windspeed,
            "day": date.day,
            "month": date.month,
            "year": date.year
        }

        return pd.DataFrame(data, index=[0])

    new_data = user_input_parameters()

    with st.expander("🔍 See model input"):
        st.dataframe(new_data)

    # ---------- PREDICTION BUTTON ----------
    if st.button("🔮 Predict Bike Demand"):
        prediction = model.predict(new_data)
        predicted_value = int(prediction[0])

        st.success(f"🚴 Estimated Bike Rentals: {predicted_value}")

        # ---------- VISUALIZATION ----------
        st.subheader("📊 Prediction Visualization (KDE Plot)")

        kde_data = pd.Series(
            [predicted_value * (1 + i / 100) for i in range(-10, 11)]
        )

        fig, ax = plt.subplots()
        sns.kdeplot(kde_data, fill=True, ax=ax)

        ax.set_title("KDE Plot of Predicted Bike Rentals")
        ax.set_xlabel("Bike Rentals")
        ax.set_ylabel("Density")

        st.pyplot(fig)

# ==========================================
# ========== CSV / EXCEL UPLOAD ==============
# ==========================================
else:
    st.subheader("Upload CSV or Excel File")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        # Detect file type
        if uploaded_file.name.endswith(".csv"):
            input_df = pd.read_csv(uploaded_file)
        else:
            input_df = pd.read_excel(uploaded_file)

        st.write("📄 Uploaded Data")
        st.dataframe(input_df)

        try:
            predictions = model.predict(input_df)
            input_df["Predicted_Bike_Rentals"] = predictions.astype(int)

            st.success("✅ Prediction Completed")
            st.dataframe(input_df)

            # Download results as CSV
            csv = input_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Prediction CSV",
                data=csv,
                file_name="bike_demand_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("❌ File format does not match model input")
            st.write(str(e))

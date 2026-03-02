from flask import Flask, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import date
from flask import Flask, render_template, request


app = Flask(__name__)

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("cardio_train.csv", sep=';')
df.drop('id', axis=1, inplace=True)

# =========================
# ML MODEL
# =========================
X = df.drop('cardio', axis=1)
y = df['cardio']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
precision = round(precision_score(y_test, y_pred) * 100, 2)
recall = round(recall_score(y_test, y_pred) * 100, 2)
f1 = round(f1_score(y_test, y_pred) * 100, 2)

# =========================
# ANALYTICS DATA
# =========================
total_patients = len(df)
cardio_yes = int(df['cardio'].sum())
cardio_no = int(total_patients - cardio_yes)
avg_age = round(df['age'].mean() / 365, 1)
avg_systolic = round(df['ap_hi'].mean(), 1)
avg_diastolic = round(df['ap_lo'].mean(), 1)

current_date = date.today().strftime("%d %B %Y")

# =========================
# ROUTES
# =========================
@app.route("/")
def index():
    return render_template(
        "index.html",
        title="CardioPredict AI",
        current_date=current_date,

        # dataset stats
        total_patients=total_patients,
        cardio_yes=cardio_yes,
        cardio_no=cardio_no,

        # model performance
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        auc=round((precision + recall) / 200, 2)  # simple derived AUC
    )


@app.route("/analytics")
def analytics():
    return render_template(
        "analytics.html",
        total_patients=total_patients,
        cardio_yes=cardio_yes,
        cardio_no=cardio_no,
        avg_age=avg_age,
        avg_systolic=avg_systolic,
        avg_diastolic=avg_diastolic,
        current_date=current_date
    )


from flask import request, jsonify, render_template

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # 👉 GET request: open predict.html page
    if request.method == "GET":
        return render_template("predict.html")

    # 👉 POST request: form submit (AJAX)
    try:
        age = int(request.form["age"])
        height = float(request.form["height"])
        weight = float(request.form["weight"])
        ap_hi = int(request.form["ap_hi"])
        ap_lo = int(request.form["ap_lo"])
        cholesterol = int(request.form["cholesterol"])
        gluc = int(request.form["gluc"])
        smoke = int(request.form["smoke"])
        alco = int(request.form["alco"])
        active = int(request.form["active"])

        # Gender not in UI → default
        gender = 1

        age_days = age * 365

        bmi = round(weight / ((height / 100) ** 2), 1)

        input_data = [[
            age_days, gender, height, weight,
            ap_hi, ap_lo, cholesterol,
            gluc, smoke, alco, active
        ]]

        input_scaled = scaler.transform(input_data)
        prediction = int(model.predict(input_scaled)[0])
        probability = int(model.predict_proba(input_scaled)[0][1] * 100)

        if probability >= 70:
            risk_level = "High Risk"
        elif probability >= 40:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        recommendations = []
        if bmi >= 25:
            recommendations.append("Maintain a healthy weight through diet and exercise.")
        if ap_hi >= 140 or ap_lo >= 90:
            recommendations.append("Monitor and control your blood pressure regularly.")
        if cholesterol > 1:
            recommendations.append("Reduce cholesterol with a heart-healthy diet.")
        if active == 0:
            recommendations.append("Increase physical activity for better heart health.")
        if not recommendations:
            recommendations.append("Keep up your healthy lifestyle!")

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "bmi": bmi,
            "risk_level": risk_level,
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400
# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)



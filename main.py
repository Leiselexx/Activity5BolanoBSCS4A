from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

model = joblib.load('svm_model-2.pkl')

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        age = float(request.form['age'])
        income = float(request.form['income'])
        family = float(request.form['family'])
        ccavg = float(request.form['ccavg'])
        mortgage = float(request.form['mortgage'])
        personalLoan = float(request.form['personalLoan'])
        cdaccount = float(request.form['cdaccount'])

        # Make a prediction using the loaded model
        prediction = model.predict([[age, income, family, ccavg, mortgage, personalLoan,cdaccount]])[0]

        prediction = int(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

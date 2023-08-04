from flask import Flask,render_template,request
import numpy as np
import pickle
import pandas as pd
import uuid
Model=pickle.load(open(r"C:\Users\lenovo\Intenship Project\flask\thyroid1_model.pkl",'rb'))
le=pickle.load(open("label_encoder.pkl",'rb'))
app=Flask(__name__)
@app.route("/")
def about():
    return render_template('home.html')

@app.route('/home', methods=['POST','GET'])
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        try:
            x = [[float(x) for x in request.form.values()]]
            col = ['goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG']
            x_df = pd.DataFrame(x, columns=col)

            # Make prediction using the loaded model
            pred = Model.predict(x_df)
            pred_label = le.inverse_transform(pred)[0]

            # Generate a unique request ID
            request_id = str(uuid.uuid4())

            return render_template('submit.html', prediction_text=pred_label, request_id=request_id)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            return render_template('submit.html', error_message=error_message)

    return render_template('submit.html', prediction_text="", request_id="")

if __name__ =="__main__":
    app.run(debug=False)
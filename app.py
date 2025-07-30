from flask import Flask,render_template,request
import pickle
import numpy as np
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def main():
    return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict():
    float_feauture = [float(x) for x in request.form.values()]
    features = [np.array(float_feauture)]
    prediction = model.predict(features)
    return render_template('index.html', prediction_text="The Prediction is: {}".format(prediction))

if __name__ == '__main__':
    app.run(debug=True)


import numpy as np
from flask import Flask , request,render_template,url_for
import pickle

app = Flask(__name__)
model = pickle.load(open('Depression.pkl' , 'rb'))
tf = pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict' , methods=['Post'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = tf.transform(data).toarray()
    	my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
import numpy as np
from flask import Flask, request, render_template
import pickle
from train import recommend_items

app= Flask(__name__)

# model= pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    int_features=[int(x) for x in request.form.values()]
    features=[np.array(int_features)]
    userId=int_features[0]
    print(userId)
    num=int_features[1]
    print(num)
    prediction= recommend_items(userID=userId, num_recommendations= num)
    print(prediction)
    # return render_template('index.html', prediction_txt= (prediction))
    return render_template('index.html',  tables=[prediction.to_html(classes='data')], titles=prediction.columns.values)

if __name__=="__main__":
    app.run()
    
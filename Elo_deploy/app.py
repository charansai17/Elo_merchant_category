from flask import Flask, jsonify, request,render_template
#from sklearn.externals import joblib
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)


cols = ['feature_2', 'feature_3', 'category_1_x', 'category_3_x', 'month_lag_x',
       'category_2_x', 'category_1_y', 'category_3_y', 'month_lag_y',
       'state_id_y', 'subsector_id_y', 'merch_month', 'merch_day',
       'merch_month_diff', 'new_month', 'new_day', 'new_weekday' ]

def root_mean_squared_error(y_true, y_pred):
    """Root mean squared error regression loss"""
    return np.sqrt(np.mean(np.square(y_true-y_pred)))

###################################################

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    model = joblib.load('lasso_regression.pkl')
    df = pd.read_csv('data.csv', index_col=0)
    y = df["target"].values
    X = df.drop(labels="target",axis=1)
    
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]       

    prediction = model.predict(final_features)
    output = prediction[0]

    rmse_train = root_mean_squared_error(np.mean(y), prediction)
    print(rmse_train)
   
    #return jsonify({'prediction': prediction})
    return render_template('form.html', result=output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

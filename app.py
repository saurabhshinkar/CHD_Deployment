
import flask
from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import PredictPipeline
from src.utils import get_data_from_sql

application=Flask(__name__)

app=application
############################################################################################
from flask import request, jsonify

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    # Get the ID from the request
    id = request.json.get('id')

    # Retrieve data from SQL based on the provided ID
    # Assuming you have a function to fetch data from SQL, replace 'getfromsql' with your function
    pred_df = get_data_from_sql(id)
    #get_data_from_sql(3372)

    if pred_df.empty:
        return jsonify({'error': 'No data found for the provided ID'}), 404

    print(pred_df)
    print("Before Prediction")

    # Assuming PredictPipeline is a class that handles your prediction pipeline
    predict_pipeline = PredictPipeline()
    print("Mid Prediction")
    results = predict_pipeline.predict(pred_df)
    print("After Prediction")

    # Convert results to JSON format
    results_df = pd.DataFrame({'prediction': results})
    results_json = results_df.to_json(orient='records')
    d={'Prediction':results_df['prediction'][0]}

    return jsonify(d)
###########################################################################################
# ## Route for a home page

# @app.route('/')
# def index():
#     return render_template('index.html') 


# @app.route('/predictdata',methods=['GET','POST'])
# def predict_datapoint():
#     if request.method=='GET':
#         return render_template('home.html')
#     else:
#         # data=CustomData(
#         #     gender=request.form.get('gender'),
#         #     race_ethnicity=request.form.get('ethnicity'),
#         #     parental_level_of_education=request.form.get('parental_level_of_education'),
#         #     lunch=request.form.get('lunch'),
#         #     test_preparation_course=request.form.get('test_preparation_course'),
#         #     reading_score=float(request.form.get('writing_score')),
#         #     writing_score=float(request.form.get('reading_score')) )

#         pred_df=data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         predict_pipeline=PredictPipeline()
#         print("Mid Prediction")
#         results=predict_pipeline.predict(pred_df)
#         print("after Prediction")
#         return render_template('home.html',results=results[0])
    

if __name__=="__main__":    
    app.run(debug=True)
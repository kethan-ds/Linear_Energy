from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predicted',methods=['POST'])
def predict_datapoint():
        data=CustomData(
                building_type = request.form.get('building_type'),
                square_footage = int(request.form.get('square_footage')),
                number_of_occupants = int(request.form.get('number_of_occupants')),
                appliances_used = int(request.form.get('appliances_used')),
                average_temperature = float(request.form.get('average_temperature')),
                day_of_week = request.form.get('day_of_week')

            )
        final_new_data=data.get_data_as_dataframe()
        y=final_new_data['Building Type'].values[0]
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)
    
        results=round(pred[0],1)
        #results=pred
    
        return render_template('results.html',final_result=[results,y])


@app.route('/topredict',methods=['GET'])
def topredict():
    return render_template('form.html')
    
    
    




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)



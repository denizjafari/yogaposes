from yogaposes.pipeline.prediction import PosePrediction
from yogaposes.utils.common import decodeImage
from flask_cors import CORS, cross_origin
import os
from flask import Flask, request, render_template


# set environment variables 
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

class YogaPoseApp:
    def __init__(self):
        self.filename = 'sample-image.jpg'
        self.classifier = PosePrediction()
        

# you wanna get sth from the app
@app.route('/', methods=['GET'])
@cross_origin()
# rendering the index html under the template folder
def home():
    return render_template('index.html')    

@app.route('/train', methods=['GET','POST'])
@cross_origin()
def trainRoute():
    os.system('dvc repro')
    return 'Training done successfully!'


@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    prediction = clApp.classifier.predict(clApp.filename)
    return prediction

if __name__ == '__main__':
    clApp = YogaPoseApp()
    app.run(host='0.0.0.0', port=8000)


# if __name__ == "__main__":
#     try:
#         prediction = PosePrediction()
#         image_path = input("Enter Image File Path: ")
#         c= prediction.predict(image_path)
#         prediction.display_image(image_path)
#         print('The predicted pose is: ', c)
#     except Exception as e:
#         raise e
        
        
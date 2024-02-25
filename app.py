"""
This is app.py
This is where the API source code written.

We have two main routes:

- /home: this is the testing route
- /getPrediction: this route will return the prediction from our ML Model
"""

# imports are written here
import logging
from utils import predict, preprocessing
from flask import Flask, jsonify, request

# setting up flask
app = Flask(__name__)

# setting up logging
logging.basicConfig(filename='app.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# first testing route
@app.route('/home', methods=['GET'])
def testing_route():
    logging.info(f"Testing Route Hit..!")
    return jsonify({'success': 1,
                    "data": "Endpoint is working..!"}), 200


# main route
@app.route('/getPrediction', methods=['POST'])
def getPrediction():
    try:
        # fetch data from API
        data = request.get_json()

        logging.info(f"Data received: {data}")

        # preprocess data
        pre_data = preprocessing(data)

        # get prediction
        prediction = predict(pre_data)

        if prediction is not None:
            return jsonify({"success": 1,
                            "prediction": str(prediction)}), 200
        else:
            return jsonify({"success": 0,
                            "error": "Internal Server Error"}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# run script code
if __name__ == '__main__':
    app.run(
        debug=False,
        host='0.0.0.0',
        port=5000,
    )
    logging.info(f"API Started..!")

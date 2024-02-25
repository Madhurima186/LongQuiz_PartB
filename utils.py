"""
This is the utils module where some utility functions are written.
"""

# Imports are written here
import joblib
import logging
import numpy as np

# logging
logging.basicConfig(filename='app.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Loading the model
model = joblib.load('logisticRegressionModel.joblib')


def preprocessing(data) -> dict:
    """
    Function to preprocess the data turn necessary categorical data to numeric.
    :param data: dict of data containing categorical values
    :return: dict of preprocessed data
    """
    try:
        logging.info('Preprocessing')
        gender_mapping = {"female": 0, "male": 1}
        chest_pain_mapping = {'typical': 1, 'nontypical': 2, 'nonanginal': 3, 'asymptomatic': 4}
        thal_mapping = {'normal': 1, 'fixed': 2, 'reversable': 3}

        data['ChestPain'] = chest_pain_mapping.get(data.get('ChestPain', '').lower(), data.get('ChestPain'))
        data['Sex'] = gender_mapping.get(data.get('Sex', '').lower(), data.get('Sex'))
        data['Thal'] = thal_mapping.get(data.get('Thal', '').lower(), data.get('Thal'))

        return data
    except Exception as e:
        logging.error(f"Error in preprocessing: ", e)


def predict(pre_data) -> bool:
    """
    Function to predict the outcome of the heart disease
    :param pre_data:
    :return: boolean indicating whether patient has heart disease or not.
    """
    try:
        # list values in pre_data
        values = list(pre_data.values())

        # Convert values to numpy array
        array = np.array(values)

        # Reshape the array
        array = array.reshape(1, -1)

        # predicting on new data
        prediction = model.predict(array)

        logging.info(f"prediction: {prediction[0]} and response: {prediction[0] == 1}")

        return prediction[0] == 1

    except Exception as e:
        logging.error(f"Error in prediction: {e}")

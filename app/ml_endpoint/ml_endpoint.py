from pathlib import Path

# Flask stuff
import flask
import flasgger.utils

# Iris classifier class
from app.ml_models.model_classes.iris_classifier import IrisClassifier

serialized_models_path = Path('app/ml_models/serialized_models/')

# Deserialize model
model = IrisClassifier().read_model(str(serialized_models_path / 'model.pickle'))

# Blueprint for the endpoint
ml_endpoint_bp = flask.Blueprint("ml_endpoint", __name__)


@ml_endpoint_bp.route("/ml_endpoint", methods=["POST"])
@flasgger.utils.swag_from("swagger/ml_endpoint.yaml", validation=True)
def example_endpoint():
    # Get posted input from api call    
    inputs = flask.request.get_json()
    # Make single prediction
    prediction = model.single_prediction(
        sepal_length=float(inputs['sepal_length']),
        sepal_width=float(inputs['sepal_width']),
        petal_length=float(inputs['petal_length']),
        petal_width=float(inputs['petal_width'])
    )
    # Return prediction
    response = {'species_prediction': prediction}
    response_json = flask.jsonify(response)
    return response_json
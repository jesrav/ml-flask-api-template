from pathlib import Path

# Flask stuff
from flask import jsonify, request, Blueprint
from flasgger.utils import swag_from

# Iris claassifier class
from app.ml_models.model_classes.iris_classifier import IrisClassifier

serialized_models_path = Path('app/ml_models/serialized_models/')

# Desirialize model
model = IrisClassifier().read_model(str(serialized_models_path / 'model.pickle'))

# Blueprint for the endpoint
ml_endpoint_bp = Blueprint("ml_endpoint", __name__)

@ml_endpoint_bp.route("/ml_endpoint", methods=["POST"])
@swag_from("swagger/ml_endpoint.yaml", validation=True)
def example_endpoint():
    # Get posted input from api call    
    inputs = request.get_json()
    prediction = model.single_prediction(
        sepal_length=float(inputs['sepal_length']),
        sepal_width=float(inputs['sepal_width']),
        petal_length=float(inputs['petal_length']),
        petal_width=float(inputs['petal_width'])
    )
    response = {'species_prediction': prediction}
    response_json = jsonify(response)
    return response_json
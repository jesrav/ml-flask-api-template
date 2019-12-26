from pathlib import Path

# Flask stuff
import flask
import flasgger.utils

# Iris classifier class
from app.ml_models.model_classes.iris_classifier import RandomForestClassifierModel

serialized_models_path = Path('app/ml_models/serialized_models/')

# Deserialize model
model = RandomForestClassifierModel().load(str(serialized_models_path / 'model.pickle'))

# Blueprint for the endpoint
ml_endpoint_bp = flask.Blueprint("predict", __name__)


@ml_endpoint_bp.route("/predict", methods=["POST"])
@flasgger.utils.swag_from("swagger/openapi_specification.yaml", validation=True)
def example_endpoint():
    # Get posted input from api call    
    json_data = flask.request.get_json()
    print(json_data)
    predictions = model.predict(model.record_dict_to_model_input(json_data))
    # Return prediction
    response = {'predictions': predictions.tolist()}
    response_json = flask.jsonify(response)
    return response_json

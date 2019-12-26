import pickle as pickle
from datetime import datetime
from abc import ABCMeta, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from marshmallow_dataframe import RecordsDataFrameSchema
import json
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin


class BaseModel(metaclass=ABCMeta):
    """
    Base class for models

    The class has a save and load method for serializing model objects.
    It enforces implementation of a fit and predict method and a model name attribute.
    """
    def __init__(self):
        self.model_initiated_dt = datetime.utcnow()

    @property
    @classmethod
    @abstractmethod
    def MODEL_NAME(self):
        pass

    # @property
    # @abstractmethod
    # def input_schema(self):
    #     raise
    #
    # @property
    # @abstractmethod
    # def output_schema(self):
    #     raise

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def __str__(self):
        return f'Model: {self.MODEL_NAME},  initiated at {self.model_initiated_dt}'

    def save(self, **kwargs):
        """Serialize model to file or variable
        """
        serialize_dict = self.__dict__

        if "fname" in kwargs.keys():
            fname = kwargs["fname"]
            with open(fname, "wb") as f:
                pickle.dump(serialize_dict, f)
        else:
            pickled = pickle.dumps(serialize_dict)
            return pickled

    def load(self, serialized):
        """Deserialize model from file or variable"""
        assert isinstance(serialized, str) or isinstance(
            serialized, bytes
        ), "serialized must be a string (filepath) or a bytes object with the serialized model"
        model = self.__class__()

        if isinstance(serialized, str):
            with open(serialized, "rb") as f:
                serialize_dict = pickle.load(f)
        else:
            serialize_dict = pickle.loads(serialized)

        # Set attributes of model
        model.__dict__ = serialize_dict

        return model


class RandomForestClassifierModel(BaseModel):
    MODEL_NAME = 'Random forest model'

    def __init__(
            self,
            features=None,
            input_dtypes=None,
            random_forest_params={'n_estimators': 100, 'max_depth': 30}
    ):
        super().__init__()
        self.features = features
        self.input_dtypes = None
        self.random_forest_params = random_forest_params
        self.model = RandomForestClassifier(**random_forest_params)

    def fit(self, X, y):
        if self.input_dtypes is None:
            self.input_dtypes = X[self.features].dtypes
        self.model.fit(X[self.features], y)
        return self.model

    def predict(self, X):
        # assert all(list(X.columns) == self.features), f'The following features must be in X: {self.features}'
        assert X[self.features].dtypes.to_dict() == self.input_dtypes.to_dict(), f'Dtypes must be: {self.input_dtypes.to_dict()}'
        predictions = self.model.predict(X[self.features])
        return predictions

    def get_model_input_schema(self):
        class ModelInputSchema(RecordsDataFrameSchema):
            """Automatically generated schema for model input dataframe"""
            class Meta:
                dtypes = self.input_dtypes
        return ModelInputSchema

    def record_dict_to_model_input(self, dict_data):
        model_input_schema = self.get_model_input_schema()()
        return model_input_schema.load(dict_data)

    def get_api_spec(self):
        # Create an APISpec
        spec = APISpec(
            title="Prediction open api spec",
            version="1.0.0",
            openapi_version="3.0.2",
            plugins=[MarshmallowPlugin()],
        )
        ModelInputSchema = self.get_model_input_schema()
        spec.components.schema("predict", schema=ModelInputSchema)
        spec.path(
            path="/predict/",
            operations=dict(
                post=dict(
                    responses={"200": {"content": {"application/json": {"schema": "ModelInputSchema"}}}}
                )
            ),
        )
        return spec
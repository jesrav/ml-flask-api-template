from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import pickle as pickle
import pandas as pd

class IrisClassifier(object):
    '''
    Iris classifier. 
    Wraps the sklearn k-neares kneighbors class in order to:
    - Have fit and train methods take data frames and validate feature names.
    - Have method for predicting single instance, to be used in api.
    '''
    def __init__(self ,n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.target = 'species'
        self.features = [
            'sepal_length', 
            'sepal_width', 
            'petal_length', 
            'petal_width'
        ]
        self.classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.label_encoder_species = LabelEncoder()
        
    def fit(self, X, y):
        """Fit the iris classification model

        Parameters
        ----------
        X : Data frame with required features:
            "sepal_length": float
            "sepal_width": float
            "petal_length": float
            "petal_width": float
            
        y : Pandas series of type str, containing the species.

        Returns
        -------
        self : object
            Returns self.
        """
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert all(
            [col in X.columns for col in self.features]
        ), "X must contain the features {}".format(self.features)
        assert (
            isinstance(y, pd.Series) and y.dtypes == "object"
        ), "Y must be a Pandas series, of type string"
        assert X.shape[0] == y.shape[0], "X and y must have same number of rows"

        X_copy = X.copy()
        y_copy = y.copy()

        # Sort columns
        X_copy = X_copy[self.features]

        # Fit label encoder for species
        y_copy = self.label_encoder_species.fit_transform(y_copy)

        # fit classifier
        self.classifier.fit(X_copy, y_copy)

        return self

    def predict(self, X):
        """Predict the species for the provided data

        Parameters
        ----------
        X : Data frame with required features:
            "sepal_length": float
            "sepal_width": float
            "petal_length": float
            "petal_width": float
            
        Returns
        -------
        y : Pandas series of type str
            Predicted species
        """
        assert isinstance(X, pd.DataFrame), "X must be a Pandas data frame"
        assert all(
            [col in X.columns for col in self.features]
        ), "X must contain the features {}".format(self.features)
        
        X_copy = X.copy()

        # Sort columns
        X_copy = X_copy[self.features]

        # Predict
        encoded_predictions = self.classifier.predict(X_copy)

        # Transform predictions to species label
        predictions = self.label_encoder_species.inverse_transform(
            encoded_predictions
        )
        return predictions

    def single_prediction(
        self, 
        sepal_length, 
        sepal_width, 
        petal_length, 
        petal_width
    ):
        """Predict a single target variable

        Parameters
        ----------
        sepal_length : float
            Sepal length
        sepal_width : float
            Sepal width
        petal_length : float
            Petal length
        petal_width : float
            Petal width
      
        Returns
        -------
        y : str
            Predicted species
        """
        assert isinstance(sepal_length, float), "col_latitude must be a float"
        assert isinstance(sepal_width, float), "col_latitude must be a float"
        assert isinstance(petal_length, float), "col_latitude must be a float"
        assert isinstance(petal_width, float), "col_latitude must be a float"
      
        # Set input parameters as columns in a data frame
        df_input = pd.DataFrame([[
                sepal_length, 
                sepal_width, 
                petal_length, 
                petal_width
            ]],
            columns=self.features,
        )

        return self.predict(df_input)[0]

    def write_model(self, **kwargs):
        """Serialize model to file or variable
        """
        serialize_dict = {
            "n_neighbors": self.n_neighbors,
            "classifier": self.classifier,
            "label_encoder_species": self.label_encoder_species,
        }

        if "fname" in kwargs.keys():
            fname = kwargs["fname"]
            with open(fname, "wb") as f:
                pickle.dump(serialize_dict, f)
        else:
            pickled = pickle.dumps(serialize_dict)
            return pickled

    @staticmethod
    def read_model(serialized):
        """Deserialize model from file or variable
        """
        assert isinstance(serialized, str) or isinstance(
            serialized, bytes
        ), "serialized must be a string (filepath) or a bytes object with the serialized model"
        model = IrisClassifier()

        if isinstance(serialized, str):
            with open(serialized, "rb") as f:
                serialize_dict = pickle.load(f)
        else:
            serialize_dict = pickle.loads(serialized)

        model.n_neighbors = serialize_dict["n_neighbors"]
        model.classifier = serialize_dict["classifier"]
        model.label_encoder_species = serialize_dict[
            "label_encoder_species"
        ]
        return model


import os
import logging
from flask import Flask
from flasgger import Swagger

from config import get_config

swagger = Swagger()


def create_app(enviroment):
    logging.info("Init app")
    app = Flask(__name__)

    logging.info("Load configuration")
    app.config.from_object(get_config(enviroment))
    logging.info(f"App running in {enviroment}")

    logging.info("Initialize swaggerui") 
    swagger.init_app(app)

    logging.info("Register ml endpoint")
    from app.ml_endpoint import bp as ml_endpoint_bp
    app.register_blueprint(ml_endpoint_bp)

    logging.info("Running app")

    return app

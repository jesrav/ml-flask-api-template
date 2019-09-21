import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    DEBUG = False
    SWAGGER = {
        "title": "ML-API",
        "uiversion": 3,
        "consumes": ["application/json"],
        "produces": ["application/json"],
    }


class Development(Config):
    ENVIRONMENT = 'development'
    DEBUG = True


class Staging(Config):
    ENVIROMENT = 'staging'
    DEBUG = False


class Production(Config):
    ENVIROMENT = 'production'
    DEBUG = False


def get_config(enviroment):
    if enviroment == "production":
        return Production
    elif enviroment == "staging":
        return Staging
    else:
        return Development

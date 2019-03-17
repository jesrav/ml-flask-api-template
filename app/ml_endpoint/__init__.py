from flask import Blueprint

bp = Blueprint("ml_endpoint", __name__)

from app.ml_endpoint import ml_endpoint  # noqa: E402 F401

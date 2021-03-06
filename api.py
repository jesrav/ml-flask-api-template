import os
import logging

from app import create_app

logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
) 

app = create_app(os.getenv("ENVIRONMENT"))

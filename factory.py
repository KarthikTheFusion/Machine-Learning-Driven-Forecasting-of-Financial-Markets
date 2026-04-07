import os
import warnings

from flask import Flask

from .routes import register_routes


warnings.filterwarnings("ignore")


def create_app():
    base_dir=os.path.dirname(os.path.dirname(__file__))
    app=Flask(__name__,static_folder=os.path.join(base_dir,"static"))
    register_routes(app)
    return app

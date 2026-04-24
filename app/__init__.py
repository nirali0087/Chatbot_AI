from flask import Flask
from flask_login import LoginManager
from flask_migrate import Migrate
from .config import Config
from .models import db, User
from .controllers import auth_bp, chat_bp
from .services import initialize_vector_store
from app.services.vector_utils import initialize_llm
from dotenv import load_dotenv
import os

login_manager = LoginManager()
migrate = Migrate()

def create_app():
    app = Flask(__name__, 
                template_folder='views/templates',  
                static_folder='views/static') 
    app.config.from_object(Config)
    
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    if Config.CONTEXT_ENABLE:
        app.vector_store = initialize_vector_store()
    else:
        app.vector_store = None
    
    app.llm = initialize_llm()
    
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(chat_bp)
    
    with app.app_context():
        db.create_all()
    
    return app
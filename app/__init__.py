from .app import Flask

from app.libs.error import APIException
from app.libs.error_code import ServerError
from flask_login import LoginManager
from flask_mail import Mail

login_manager = LoginManager()
mail = Mail()


def register_blueprint(app):
    from app.web.book import web
    from app.web.adversarial_example import AD
    from app.api.v1 import create_blueprint_v1
    app.register_blueprint(web)
    app.register_blueprint(AD)
    app.register_blueprint(create_blueprint_v1(), url_prefix='/v1')


def register_plugin(app):
    login_manager.init_app(app)
    login_manager.login_view = 'web.login'
    login_manager.login_message = '请先登录或注册'

    mail.init_app(app)
    from app.models.base import db
    db.init_app(app)
    with app.app_context():
        db.create_all()


def create_app():
    app = Flask(__name__)
    app.config.from_object('app.secure')  # 载入配置文件
    app.config.from_object('app.setting')  # 载入配置文件

    register_blueprint(app)
    register_plugin(app)

    return app

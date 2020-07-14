from werkzeug.exceptions import HTTPException
from app.libs.error import APIException
from app.libs.error_code import ServerError

from app import create_app

app = create_app()


@app.errorhandler(Exception)
def framework_error(e):
    if isinstance(e, APIException):
        return e
    if isinstance(e, HTTPException):
        code = e.code
        msg = e.description
        error_code = 1007
        return APIException(msg, code, error_code)
    else:
        # 调试模式
        # log
        if not app.config['DEBUG']:
            return ServerError()
        else:
            raise e


if __name__ == '__main__':
    # 生产环境 nginx + uwsgi
    app.run(host='0.0.0.0', debug=app.config['DEBUG'], port=5000, threaded=True)

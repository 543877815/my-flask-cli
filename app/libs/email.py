from threading import Thread

from flask import current_app, render_template

from app import mail
from flask_mail import Message


def send_async_mail(app, msg):
    with app.app_context():
        try:
            mail.send(msg)
        except Exception as e:
            pass


def send_mail(to, subject, template, **kwargs):
    # msg = Message('测试邮件', sender='543877815@qq.com', body='Test', recipients=['543877815@qq.com'])
    msg = Message('[鱼书]' + '' + subject,
                  sender=current_app.config['MAIL_USERNAME'],
                  recipients=[to])
    msg.html = render_template(template, **kwargs)
    app = current_app._get_current_object()
    thr = Thread(target=send_async_mail, args=[app, msg])
    thr.start()

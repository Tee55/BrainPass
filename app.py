import os
import string
import json
from flask import Flask, request, redirect, render_template, request, url_for, flash, jsonify, send_from_directory, Response
import logging
from flask_sqlalchemy import SQLAlchemy
from flask_login import login_user, current_user, UserMixin, LoginManager, logout_user, login_required
import csv
import shutil
import pandas as pd
import random
import time
import webbrowser

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.from_object(__name__)

db = SQLAlchemy(app)

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20))

#######################################################################
# Login manager
#######################################################################
login_manager = LoginManager()
login_manager.init_app(app)
@login_manager.user_loader

#######################################################################
# User loader
#######################################################################
def user_loader(user_id):
    return User.query.get(user_id)

@app.route('/')
def index():
    db.create_all()
    if request.method == "POST":
        username = request.form['username']
        if User.query.filter_by(username=username).first() is None:
            user = User(username=username)
            db.session.add(user)
        user = User.query.filter_by(username=username).first()
        login_user(user, remember=False)
        return redirect(url_for("login"))
    else:
        return render_template('index.html')

@app.route('/instructions')
def instruction():
    return render_template('instructions.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/signup')
def signup():
    return render_template('signup.html')

##################################################################################
# Error handler
##################################################################################
@app.errorhandler(404)
def server_error(error):
    logging.exception('An error occurred during a request.')
    return """An internal error occurred: <pre>{}</pre> See logs for full stacktrace.""".format(error), 404

@app.errorhandler(500)
def server_error(error):
    logging.exception('An error occurred during a request.')
    return """An internal error occurred: <pre>{}</pre> See logs for full stacktrace.""".format(error), 500

port = int(os.environ.get('PORT', 9000))
if __name__ == '__main__':
    log_file = open("error.log", "w")
    log_file.truncate()
    log_file.close()
    logging.basicConfig(filename='error.log', level=logging.DEBUG)

    webbrowser.open('http://127.0.0.1:9000/')
    app.run(host='0.0.0.0', port=port, threaded=True)

from flask import Flask, render_template, redirect, request, url_for, flash
import argparse
from flask import jsonify
from flask_restful import Resource, Api
import serial
import struct
import time
import timeit
import atexit
import logging
import threading
import sys
import glob
import numpy as np
import pymysql
import open_bci_v3 as bci
import emotiv_epoc as emotiv
from multiprocessing import Pool

app = Flask(__name__, static_folder="static")
api = Api(app)
__author__ = 'Teerapath Sattabongkot'
__version__ = '1.0.0'

app.secret_key = '\xc4\x8f\xa9\xfe\xca \xa4\xa6K\x8f\xa8)\xb4\xdd\xa3\xf7|)F\x18\xa6\x8e\x07\xc3'


class EEG_data(threading.Thread):
    def __init__(self):
        self.data = list()
        self.board = bci.OpenBCIBoard()

    def streaming(self):
        self.data = []
        self.board.start_streaming(self.set_data)

    def set_data(self, sample):
        print(sample)
        self.data.append([sample.channel_data[0], sample.channel_data[2],
                          sample.channel_data[3], sample.channel_data[4], sample.channel_data[5]])

    def get_data(self):
        return self.data


session_1 = []
session_2 = []
session_3 = []
session_4 = []
session_5 = []

global device
global count


def set_count():
    global count
    count = 0


def initialize_board():
    if 'device' in globals():
        pass
    else:
        global device
        device = EEG_data()


class myThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, target=self.run, daemon=True)

    def run(self):
        device.streaming()


# Create new threads
thread = myThread()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about/')
def about_this():
    return render_template('about_this_project.html')


@app.route('/initialize-board/')
def initialize():
    try:
        initialize_board()
        return redirect(url_for('mobile'))
    except OSError as e:
        return render_template('device_error.html')


@app.route('/mobile-check/')
def mobile():
    return render_template('mobile_check.html')


@app.route('/mobile-get-train-1/')
def mobile_train_1():
    thread.run()
    return render_template('mobile_train.html')


@app.route('/mobile-get-train-2/')
def mobile_train_2():
    session_3.append(device.get_data())
    thread.run()
    return render_template('mobile_train.html')


@app.route('/mobile-get-train-3/')
def mobile_train_3():
    session_3.append(device.get_data())
    thread.run()
    return render_template('mobile_train.html')


@app.route('/mobile-get-train-4/')
def mobile_train_4():
    session_3.append(device.get_data())
    thread.run()
    return render_template('mobile_train.html')


@app.route('/mobile-get-train-finish/')
def mobile_train_finish():
    session_3.append(device.get_data())
    thread.__init__()
    return render_template('mobile_train_finish.html')


@app.route('/mobile-get-test/')
def mobile_run():
    thread.run()
    return render_template('mobile_test.html')


@app.route('/mobile-get-test-finish/')
def mobile_test_finish():
    session_4.append(device.get_data())
    thread.__init__()
    return render_template('mobile_test_finish.html')


@app.route('/name-signup/', methods=['GET', 'POST'])
def name_sign_in():
    if not session_1:
        pass
    else:
        del session_1[:]
    try:
        connection = pymysql.connect(host='teerapaths.ddns.net', port=3306, user='root', password='Pass_Word123456', db='brainwave')
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        username_list = [item[0] for item in cursor.fetchall()]
        connection.close()
        if request.method == 'POST':
            if request.form['submit'] == 'Submit':
                name = request.form['text']
                if name in username_list:
                    flash('Username exist!')
                    return render_template('name_signup.html')
                elif not name:
                    flash('Insert username!')
                    return render_template('name_signup.html')
                else:
                    session_1.append(name)
                    try:
                        set_count()
                        initialize_board()
                    except OSError as e:
                        return render_template('device_error.html')
                    return redirect(url_for('instruction_signup'))
            elif request.form['submit'] == 'Go back':
                return redirect(url_for('index'))
        else:
            return render_template('name_signup.html')
    except pymysql.Error as e:
        print(e)
        print("Database not connect!")
        return render_template('database_error.html')
    except serial.SerialException as e:
        return render_template('device_error.html')


@app.route('/instruction/', methods=['GET', 'POST'])
def instruction_signup():
    return render_template('instruction_signup.html')


@app.route('/signup-stimulus/', methods=['GET', 'POST'])
def sign_up():
    thread.start()
    return render_template('checkerboard_snordgrass_signup.html')


@app.route('/eye-rest/')
def sign_up_2():
    session_1.append(device.get_data())
    thread.__init__()
    global count
    count += 1
    if count >= 4:
        return render_template('evaluation.html')
    else:
        return redirect(url_for('stop_stimulus'))


@app.route('/stop-stimulus/')
def stop_stimulus():
    global count
    flash('Data collect: ' + str(count))
    return render_template('stop_stimulus.html')


@app.route('/evaluation/')
def evaluation():
    return render_template('evaluation.html')


@app.route('/sign-up-complete/')
def signupCompleted():
    return render_template('signup_complete.html')


@app.route('/name-login/', methods=['GET', 'POST'])
def name_login():
    if not session_2:
        pass
    else:
        del session_2[:]
    try:
        connection = pymysql.connect(host='teerapaths.ddns.net', port=3306, user='root', password='Pass_Word123456', db='brainwave')
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        username_list = [item[0] for item in cursor.fetchall()]
        connection.close()
        if request.method == 'POST':
            if request.form['submit'] == 'Submit':
                name = request.form['text']
                print("We are here")
                if not name in username_list:
                    flash('Username does not exist!')
                    return render_template('name_login.html')
                elif not name:
                    flash('Insert username!')
                    return render_template('name_login.html')
                else:
                    session_2.append(name)
                    try:
                        set_count()
                        initialize_board()
                        return redirect(url_for('instruction_login'))
                    except OSError as e:
                        return render_template('device_error.html')
                    except pymysql.Error as e:
                        print(e)
                        print("Database not connect!")
                        return render_template('database_error.html')

            elif request.form['submit'] == 'Go back':
                return redirect(url_for('index'))
        else:
            return render_template('name_login.html')
    except serial.SerialException as e:
        return render_template('device_error.html')


@app.route('/instruction-login/', methods=['GET', 'POST'])
def instruction_login():
    return render_template('instruction_login.html')


@app.route('/login-stimulus/')
def login():
    thread.start()
    global start_com
    start_com = time.time()
    return render_template('checkerboard_snordgrass_login.html')


@app.route('/processing/')
def processing():
    session_2.append(device.get_data())
    thread.__init__()
    return render_template('processing.html')


@app.route('/login-complete/')
def login_complete():
    flash(session_2[0], category="username")
    present_time = time.strftime('%X %x')
    com_time = time.time() - start_com
    flash(present_time, category="time")
    flash(com_time, category="time_com")
    return render_template('information.html')


@app.route('/login-failed/')
def login_failed():
    return render_template('login_failed.html')


@app.route('/improve-model/', methods=['POST', 'GET'])
def improve():
    thread.start()
    return render_template('checkerboard_snordgrass_improve.html')


@app.route('/improve-finish/')
def improve_finish():
    session_5.append(session_2[0])
    session_5.append(device.get_data())
    thread.__init__()
    return render_template('improve_finish.html')


# Error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html')


class dataTrain(Resource):
    def get(self):
        return jsonify(session_1)


class dataTest(Resource):
    def get(self):
        return jsonify(session_2)


class dataTrain_mobile(Resource):
    def get(self):
        return jsonify(session_3)


class dataTest_mobile(Resource):
    def get(self):
        return jsonify(session_4)


class dataImprove(Resource):
    def get(self):
        return jsonify(session_5)


api.add_resource(dataTrain, '/datatrain')
api.add_resource(dataTest, '/datatest')

api.add_resource(dataImprove, '/dataimprove')

api.add_resource(dataTrain_mobile, '/datatrain_mobile')
api.add_resource(dataTest_mobile, '/datatest_mobile')

if __name__ == '__main__':
    import webbrowser

    webbrowser.open('http://127.0.0.1:9000/')
    app.run(host='127.0.0.1', port=9000, threaded=True)

from flask import Flask
from flask import jsonify
from flask import request
from inceptionV3 import classifier_inceptionV3

import os
import psutil


app = Flask(__name__)
app.register_blueprint(classifier_inceptionV3, url_prefix="/classifier_inceptionV3")


print("\n\n\n\n")
print("*** App is loaded")
print("\n\n\n\n")


@app.route("/", methods=['GET', 'POST'])
def home():
    return jsonify("Working")


@app.route("/info", methods=['GET', 'POST'])
def info():

    process = psutil.Process(os.getpid())
    print("\n\nUsage: ")
    print(format_bytes(process.memory_info().rss))

    usage = format_bytes(process.memory_info().rss)

    return jsonify(str(round(usage[0], 2)) + " " + usage[1])


def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'kilo', 2: 'mega', 3: 'giga', 4: 'tera'}
    while size > power:
        size /= power
        n += 1
    return size, power_labels[n]+'bytes'



from PIL import Image
import numpy as np
from flask import Flask, Response, jsonify, request, send_file
from Utils.utils import yaml_reader
import cv2
from Orchestrator.orchestrator_v3 import Orchestrator

# Flask + sqlAlchemy

app = Flask(__name__)

config_path = "C:/Users/Sumukha/Desktop/Projects/Deep_Learning_Framework/Applications/cat_vs_dog/process_input.yaml"


@app.route("/", methods=["GET"])
def welcome_page():
    return jsonify("Welcome to Cat or dog predictor")


@app.route("/predictions", methods=["POST"])
def prediction_page():
    image_raw = request.files["image"]
    image = np.array(Image.open(image_raw))
    processing_steps = yaml_reader(config_path)
    print("Starting execution")
    output_generator = Orchestrator(processing_steps)
    result = output_generator(external_inputs={"image": image})[0]
    _, result = cv2.imencode('.png', result)
    response = result.tostring()
    return Response(response=response, status=200, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5008)

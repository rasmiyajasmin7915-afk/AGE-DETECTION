from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
model = load_model("age_detection_from_face.keras")

# Class names (same order as used during training)
class_names = ["young", "middle", "old"]

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Load image
    img = image.load_img(filepath, target_size=(128, 128))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Prediction
    prediction = model.predict(img)

    print("Prediction:", prediction)

    predicted_class = np.argmax(prediction)

    print("Predicted class index:", predicted_class)

    class_names = ["young", "middle", "old"]

    predicted_label = class_names[predicted_class]

    print("Predicted label:", predicted_label)

    return render_template(
        "index.html",
        prediction=predicted_label.capitalize(),
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)
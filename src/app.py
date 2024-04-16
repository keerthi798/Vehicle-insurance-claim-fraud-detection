import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

import utils


project_dir = Path(__file__).resolve().parents[1]
app = Flask(__name__, static_url_path='/static')



with open(project_dir / "models" / "best_model.pickle", "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
   return render_template("index.html")

# @app.route("/dashboard")
# def dashboard():
#     return render_template("dashboard.html")

@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()
    preprocessed_data = utils.preprocess(form_data)

    # Add random_feature column
    preprocessed_data["random_feature"] = np.random.uniform(size=1)

    probability = model.predict_proba(pd.DataFrame(preprocessed_data, index=[0]))
    return render_template("result.html", probability=f"{probability[0, 1] * 100:.2f}")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

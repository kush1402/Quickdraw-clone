import numpy as np
import tensorflow as tf
import pickle
import os
from PIL import Image

model = tf.keras.models.load_model("my_model.h5")


with open("label_encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)


image_directory = "C:\\Users\\kunwa\\Downloads\\"
prev_image = ""
while 1:
    all_files = os.listdir(image_directory)
    image_files = [
        file for file in all_files if file.endswith(".png") and "sunny" in file
    ]

    latest_image = sorted(
        image_files,
        key=lambda file: os.path.getmtime(os.path.join(image_directory, file)),
        reverse=True,
    )[0]

    if latest_image != prev_image:
        prev_image = latest_image
        predictions = model.predict(
            np.array(
                Image.open(os.path.join(image_directory, latest_image))
                .convert("L")
                .resize((28, 28))
            ).reshape(
                (
                    1,
                    784,
                )
            )
        )

        predict_5 = np.argsort(predictions[0])
        predict_5 = predict_5 
        predict_5_string = encoder.inverse_transform(predict_5)
        top5_probabilities = predictions[0][predict_5]
        top5_predictions = [(label, prob) for label, prob in zip(predict_5_string, top5_probabilities)]
        print("Top 5 Predictions:")
        for label, prob in top5_predictions:
             print(f"Class: {label}, Probability: {prob:.4f}")

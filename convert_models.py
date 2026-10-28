from tensorflow.keras.models import load_model

# CNN
cnn_model = load_model(
    "models/pneumonia_cnn_model.keras",
    compile=False
)

cnn_model.save("models/cnn_new.keras")

print("CNN model converted successfully!")

# DenseNet
densenet_model = load_model(
    "models/densenet121_model.keras",
    compile=False
)

densenet_model.save("models/densenet_new.keras")

print("DenseNet model converted successfully!")
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.face_model import create_face_model


# Dataset path
DATASET_PATH = "dataset"

# Model save path
MODEL_PATH = "models/face_model.h5"


def train_model():

    # Image preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
    )

    train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

    


    # Create model
    model = create_face_model(
        num_classes=len(train_data.class_indices)
    )


    # Training
    model.fit(
    train_data,
    epochs=10
)

    # Save model
    os.makedirs("models", exist_ok=True)

    model.save(MODEL_PATH)

    print("Training completed and model saved!")


if __name__ == "__main__":
    train_model()
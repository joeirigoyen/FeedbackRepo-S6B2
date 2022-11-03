from pathlib import Path
import model_generator as mg


if __name__ == '__main__':
    # Set constants
    DATA_PATH = Path.cwd().parent.joinpath("data")
    TEST_PATH = Path.cwd().parent.joinpath("tests", "witch2.jpg")
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    EPOCHS = 100

    # Get classes and data
    classes = mg.get_classes_in(DATA_PATH)
    # print(classes)
    # mg.convert_images_in(DATA_PATH)
    # train, val = mg.get_datasets(DATA_PATH, image_size=(IMG_HEIGHT, IMG_WIDTH))
    # Create model
    # model = mg.get_model(len(classes), IMG_HEIGHT, IMG_WIDTH)
    # print(model.summary())
    # history = mg.fit_model(model, train, val, epochs=EPOCHS)
    # mg.get_history_plot(history, "training_3", EPOCHS, "rot 0.125, br 0.15, con 0.15")

    # Load model from previously saved file
    model = mg.get_saved_model("my_model.h5")
    result, prob = mg.get_prediction(model, TEST_PATH, (IMG_HEIGHT, IMG_WIDTH), classes)
    print(result, prob)

from pathlib import Path
import model_generator as mg


def try_again(prompt: str) -> bool:
    print(f"\n{prompt}\n")
    while True:
        try:
            user_input = False if input("Try again? (Y/n): ").lower() == "n" else True
            break
        except Exception:
            print(f"\nInvalid input.\n")
    return user_input

def get_user_path() -> str:
    while True:
        test_path = input("Please enter a path to an image you'd like to test: ")
        test_path = Path(test_path)
        if test_path.exists():
            if test_path.is_file():
                result, prob = mg.get_prediction(model, test_path, (IMG_HEIGHT, IMG_WIDTH), CLASSES)
                print(result, prob)
            else:
                if not try_again("\nThe entered path is not a file.\n"):
                    return None
        else:
            if not try_again("\nThe entered path does not exist.\n"):
                return None
        return test_path

if __name__ == '__main__':
    # Set constants
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    CLASSES = ['clown', 'frankenstein', 'mummy', 'spider', 'witch']

    # Load model from previously saved file
    model = mg.get_saved_model(Path("./Module 2/model/saved/my_model.h5"))
    while True:
        test_path = get_user_path()
        if not try_again("\n"):
            break

import sys
import glob
import numpy as np

from model import ResNetModel


def main():
    model = ResNetModel.build_model(2)
    print(model)

    train_X = []
    train_y = []
    valid_X = []
    valid_y = []
    path_list = []

    for kind in ["train", "valid"]:
        for animal in ["cat", "dog"]:
            for path in glob.glob(f"img/animals/{kind}/{animal}/*"):
                print(path)
                path_list.append(path)
                array = ResNetModel.img_to_array(path)
                print(array)
                if animal == "cat":
                    y = [1, 0]
                else:
                    y = [0, 1]
                if kind == "train":
                    train_X.append(array)
                    train_y.append(y)
                else:
                    valid_X.append(array)
                    valid_y.append(y)

    train_X = np.array(train_X)
    train_y = np.array(train_y)
    valid_X = np.array(valid_X)
    valid_y = np.array(valid_y)

    history = model.fit(train_X,
                        train_y,
                        batch_size=20,
                        epochs=10,
                        validation_data=(valid_X, valid_y))
    print(history)


if __name__ == "__main__":
    main()

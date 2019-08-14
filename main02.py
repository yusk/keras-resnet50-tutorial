import sys
import glob
import numpy as np
import math
from keras.utils import Sequence

from model import ResNetModel


class AnimalsSequence(Sequence):
    def __init__(self, kind, batch_size):
        path_list = []
        y = []
        for animal in ["cat", "dog"]:
            for path in glob.glob(f"img/animals/{kind}/{animal}/*"):
                path_list.append(path)
                y.append(animal)

        self.kind = kind
        self.batch_size = batch_size
        self.path_list = path_list
        self.y = y
        self.samples = len(path_list)

    def __getitem__(self, idx):
        X = []
        y = []
        for i in range(idx * self.batch_size,
                       idx * self.batch_size + self.batch_size):
            if i >= self.samples:
                break

            path = self.path_list[i]

            X.append(ResNetModel.img_to_array(path))
            y.append([1, 0] if self.y[i] == "cat" else [0, 1])

        X = np.array(X)
        y = np.array(y)

        return X, y

    def __len__(self):
        return math.ceil(len(self.path_list) / self.batch_size)

    def on_epoch_end(self):
        # epoch終了時の処理
        pass


def main():
    model = ResNetModel.build_model(2)

    batch_size = 2
    itr_train = AnimalsSequence("train", batch_size)
    itr_valid = AnimalsSequence("valid", batch_size)

    steps_per_epoch = math.ceil(itr_train.samples / batch_size)
    validation_steps = math.ceil(itr_valid.samples / batch_size)

    history = model.fit_generator(itr_train,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=10,
                                  validation_data=itr_valid,
                                  validation_steps=validation_steps,
                                  callbacks=[],
                                  shuffle=False)
    print(history)


if __name__ == "__main__":
    main()

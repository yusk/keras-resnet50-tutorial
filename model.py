import os
import math
import json
import pickle
import numpy as np
from datetime import datetime

from keras.models import Model
from keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.optimizers import SGD

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from keras.preprocessing.image import load_img, img_to_array

TARGET_SIZE = (224, 224)


class ResNetModel:
    @staticmethod
    def make_model_dir():
        model_dir = os.path.join('models',
                                 datetime.now().strftime('%y%m%d_%H%M'))
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    @staticmethod
    def build_model(num_classes, channel_num=3):
        index_pairs = [(92, 92), (-1, 173)]
        index = 0

        input_tensor = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1],
                                    channel_num))
        resnet_model = ResNet50(include_top=False,
                                weights='imagenet',
                                input_tensor=input_tensor)

        outputs = GlobalAveragePooling2D()(
            resnet_model.layers[index_pairs[index][0]].output)
        outputs = Dense(1024, activation='relu')(outputs)
        outputs = Dropout(0.2)(outputs)
        outputs = Dense(num_classes, activation='softmax')(outputs)

        model = Model(inputs=resnet_model.input, outputs=outputs)
        for i, layer in enumerate(model.layers):
            if i <= index_pairs[index][1] and 'BatchNormalization' not in str(
                    layer):
                layer.trainable = False
                print(i, layer, 'no train')
            else:
                print(i, layer)
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=1e-4, momentum=0.9),
                      metrics=['acc'])
        return model

    @staticmethod
    def img_to_array(img_path):
        img = load_img(img_path, target_size=TARGET_SIZE)
        img_array = preprocess_input(img_to_array(img)) / 255.
        return img_array

    def __init__(self, data_dir=None, model_path=None):
        self.data_dir = data_dir
        self.model_path = model_path

    def make_img_itr(self, batch_size):
        idg_train = ImageDataGenerator(
            rescale=1 / 255.,
            # shear_range=0.1,
            # zoom_range=0.1,
            # horizontal_flip=True,
            preprocessing_function=preprocess_input)

        img_itr_train = idg_train.flow_from_directory(os.path.join(
            self.data_dir, 'train'),
                                                      target_size=TARGET_SIZE,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

        img_itr_valid = idg_train.flow_from_directory(os.path.join(
            self.data_dir, 'valid'),
                                                      target_size=TARGET_SIZE,
                                                      batch_size=batch_size,
                                                      class_mode='categorical')

        return img_itr_train, img_itr_valid

    def make_img_itr_test(self, batch_size):
        idg_test = ImageDataGenerator(rescale=1 / 255.,
                                      preprocessing_function=preprocess_input)

        img_itr_test = idg_test.flow_from_directory(os.path.join(
            self.data_dir, 'test'),
                                                    target_size=TARGET_SIZE,
                                                    batch_size=16,
                                                    class_mode='categorical')

        return img_itr_test

    def save_data(self, model, img_itr_train):
        model_json = os.path.join(self.model_dir, 'model.json')
        with open(model_json, 'w') as f:
            json.dump(json.loads(model.to_json()), f)

        model_classes = os.path.join(self.model_dir, 'classes.pkl')
        with open(model_classes, 'wb') as f:
            pickle.dump(img_itr_train.class_indices, f)

    def make_callbacks(self):
        dir_weights = os.path.join(self.model_dir, 'weights')
        os.makedirs(dir_weights, exist_ok=True)

        es = EarlyStopping(patience=4)

        cp_filepath = os.path.join(dir_weights,
                                   'wp_{epoch: 02d}_ls_{loss: .1f}')
        cp = ModelCheckpoint(cp_filepath,
                             monitor='loss',
                             verbose=0,
                             save_best_only=False,
                             save_weights_only=True,
                             mode='auto',
                             period=7)

        csv_filepath = os.path.join(self.model_dir, 'loss.csv')
        csv = CSVLogger(csv_filepath, append=True)

        return [es, cp, csv]

    def train(self, n_epoch, batch_size):
        self.model_dir = self.make_model_dir()
        self.model_path = os.path.join(self.model_dir, 'model.h5')

        img_itr_train, img_itr_valid = self.make_img_itr(batch_size)
        model = self.build_model(len(img_itr_train.class_indices.keys()))
        print(model.summary())
        self.save_data(model, img_itr_train)
        callbacks = self.make_callbacks()

        steps_per_epoch = math.ceil(img_itr_train.samples / batch_size)
        validation_steps = math.ceil(img_itr_valid.samples / batch_size)

        history = model.fit_generator(
            img_itr_train,
            steps_per_epoch=steps_per_epoch,
            epochs=n_epoch,
            validation_data=img_itr_valid,
            validation_steps=validation_steps,
            callbacks=callbacks,
        )

        print(history)
        print(img_itr_train.class_indices)

        model.save(self.model_path)

    def test(self, batch_size):
        model = load_model(self.model_path)

        img_itr_test = self.make_img_itr_test(batch_size)

        preds = model.predict_generator(img_itr_test, steps=len(img_itr_test))
        print(preds)

        evl = model.evaluate_generator(img_itr_test, steps=len(img_itr_test))
        print(f'Test eval: {evl}')

    def test_with_file_path(self, file_path):
        model = load_model(self.model_path)

        img_array = self.img_to_array(file_path)
        print(img_array)
        print(img_array.shape)
        pred = model.predict(np.array([img_array]))[0]
        print(pred)

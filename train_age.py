import argparse
import os
import sys
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
# from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
# import keras_resnet
from keras_resnet import custom_objects
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.models import load_model

# from keras.callbacks import CSVLogger
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
HEIGHT = 196
WIDTH = 196
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))
patience = 50
TRAIN_DIR = dataset_path = os.path.abspath(os.path.join('dataset'))
input_shape = (196, 196, 3)
BATCH_SIZE = 8

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=90,
    horizontal_flip=True,
    vertical_flip=True
)

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size=(HEIGHT, WIDTH),
                                                    batch_size=BATCH_SIZE)


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)

    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    finetune_model.summary()
    return finetune_model


def main(args):
    class_list = ["0-10", "10-20", "20-30", "30-40", "40-60", "60-above"]
    FC_LAYERS = [512, 256]
    dropout = 0.5

    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    num_train_images = 41460
    adam = Adam(lr=0.00001)
    if args.model != None:
        print(args.model)
        from keras_resnet import custom_objects
        custom_objects = custom_objects.copy()
        custom_objects.update(custom_objects)
        model = load_model(args.model, custom_objects=custom_objects)
        # model = load_model(args.model)
        model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # filepath = "bin/" + "ResNet50" + "_model_weights.h5"

        checkpoint = ModelCheckpoint(filepath='trained_models/age_models/resnet50_{epoch:08d}.h5', verbose=1, save_weights_only=False,
                                     save_best_only=False, mode='max')
        csv_logger = CSVLogger('log/log.csv', append=False, separator=';')
        callbacks_list = [checkpoint, csv_logger]
        model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8,
                            steps_per_epoch=num_train_images / BATCH_SIZE, shuffle=True, callbacks=callbacks_list)
    else:
        finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS,
                                              num_classes=len(class_list))
        finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
        # filepath = "bin/" + "ResNet50" + "_model_weights.h5"

        print("HIIIIIII")

        checkpoint = ModelCheckpoint(filepath='trained_models/age_models/resnet50_{epoch:08d}.h5', verbose=1, save_weights_only=False,
                                     save_best_only=False, mode='max')
        csv_logger = CSVLogger('log/log.csv', append=False, separator=';')
        # csv_logger = CSVLogger(log_file_path, append=False)
        early_stop = EarlyStopping('val_loss', patience=patience)
        reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                      patience=int(patience / 4), verbose=1)
        callbacks = [checkpoint, csv_logger, early_stop, reduce_lr]
        # callbacks_list = [checkpoint, csv_logger]
        finetune_model.fit_generator(train_generator,
                                     #steps_per_epoch=num_train_images / BATCH_SIZE,
                                      steps_per_epoch = 5,
                                     epochs=NUM_EPOCHS,verbose=1, workers=8,
                                      shuffle=True,
                                     callbacks=callbacks)


# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    parser.add_argument('--model', help='Model to use.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

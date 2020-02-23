import argparse
import os
import sys
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.callbacks import CSVLogger
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



HEIGHT = 197
WIDTH = 197

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))



# TRAIN_DIR =     dataset_path  = os.path.abspath(os.path.join('..', 'datasets'))+'/data'
TRAIN_DIR =     dataset_path  = '/home/nishchit/majorproject/june/face_classification/datasets/UTKface_inthewild/'
# log_file_path = '../trained_models/gender_models/age_training.log'
WIDTH = 197
BATCH_SIZE = 8
# data_loader = DataManager(dataset_name)
# ground_truth_data = data_loader.get_data()
# train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)
# print('Number of training samples:', len(train_keys))
# print('Number of validation samples:', len(val_keys))
# image_generator = ImageGenerator(ground_truth_data, batch_size,
#                                  input_shape[:2],
#                                  train_keys, val_keys, None,
#                                  path_prefix=images_path,
#                                  vertical_flip_probability=0,
#                                  grayscale=grayscale,
#                                  do_random_crop=do_random_crop)

train_datagen =  ImageDataGenerator(
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

    return finetune_model



def  main(args):
    class_list = ["male","female"]
    FC_LAYERS = [1024, 1024]
    dropout = 0.5
 
    NUM_EPOCHS = 50
    BATCH_SIZE = 8
    num_train_images = 41460

    adam = Adam(lr=0.00001)   

    if args.model != None:

        print("here",args.model)

        model = load_model(args.model)
        model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

        filepath="bin/" + "ResNet50" + "_model_weights.h5"
        checkpoint = ModelCheckpoint(filepath='bin/resnet50_{epoch:08d}.h5', verbose=1, save_weights_only = False,save_best_only=False, mode='max')
        csv_logger = CSVLogger('log/log.csv', append=False, separator=';')
        callbacks_list = [checkpoint,csv_logger]

        history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True, callbacks=callbacks_list)


        plot_training(history)

    else :
   

        finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=len(class_list))


        finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

        filepath="bin/" + "ResNet50" + "_model_weights.h5"
        checkpoint = ModelCheckpoint(filepath='bin/resnet50_{epoch:08d}.h5', verbose=1, save_weights_only = False,save_best_only=False, mode='max')
        csv_logger = CSVLogger('log/log.csv', append=False, separator=';')
        callbacks_list = [checkpoint,csv_logger]

        history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True, callbacks=callbacks_list)


        plot_training(history)

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

#Command 
'''
sudo python3 transfer_image_batch.py --model bin/resnet50_00000007.h5 
'''


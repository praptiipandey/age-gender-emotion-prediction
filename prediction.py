from keras.applications.resnet50 import ResNet50, preprocess_input
import cv2
import numpy as np 
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
import os
from keras.preprocessing import image

HEIGHT = 216
WIDTH = 216

dropout = 0.5
FC_LAYERS = [512, 256] 
adam = Adam(lr=0.00001)
base_model = ResNet50(weights='imagenet',
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))
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



def load_model(weight):
    class_list = ["0-10","10-20","20-30","30-40","40-60","60-above"]


    model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list))
    model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])


    classes = [] 
    
    model.load_weights(weight)


    return model

    #
    # for file in os.listdir('imgs/'):
    #
    #     print(file)
    #
    #     img_path = 'imgs/'+file
    #     frame = cv2.imread(img_path)
    #
    #     print(frame.shape)
    #     img = image.load_img(img_path, target_size=(216,216))
    #     x = image.img_to_array(img)
    #     x = np.expand_dims(x, axis=0)
    #
    #     x = preprocess_input(x)
    #
    #     prediction_score = model.predict(x)
    #
    #     data= prediction_score[0]
    #     data= data.tolist()
    #
    #     print(data)
    #
    #     print(data.index(max(data)) )
    #     print(class_list[data.index(max(data))])
    #     # y = 20
    #
    #
    #     # resized_frame = cv2.resize(frame, (500, 500))
    #     # for i in range(len(classes)):
    #     #     print("i", i)
    #
    #
    #     #    # text = classes[i][result]
    #
    #     #     index_result = prediction_score[i].tolist()
    #
    #
    #     #     maximum_index = index_result[0].index(max(index_result[0]))
    #
    #
    #
    #
    #     #     text  = classes[i][maximum_index]
    #     #     print(text)
    #
    #
    #
    #     #     cv2.putText(resized_frame, text, (10, y),
    #     #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
    #     #         thickness=2)
    #     #     y +=20
    #
    #     # cv2.imwrite('output_img/%s'%file,resized_frame)
    #
    #
    #     # print(model.predict(x),'\n')
    #
    #     # prediction_score = model.predict(x)
    #     # print(prediction_score)

def predict(model,img,class_list):
    img = cv2.resize(img,(216,216),interpolation=cv2.INTER_AREA)
    x = np.expand_dims(img, axis=0)

    x = preprocess_input(x)

    prediction_score = model.predict(x)

    data= prediction_score[0]
    data= data.tolist()

    print(data)

    print(data.index(max(data)) )
    print(class_list[data.index(max(data))])
    return class_list[data.index(max(data))]


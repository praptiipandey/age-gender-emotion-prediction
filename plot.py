# import pandas as pd
# import matplotlib.pyplot as plt
#
# file = pd.read_csv('/home/nishchit/majorproject/august/june/face_classification/trained_models/gender_models/gender_training.log')
# plot = file.plot.line('acc')
#
#
# plt.show()

import pandas as pd
import matplotlib.pyplot as plt
file = pd.read_csv('/home/nishchit/majorproject/august/june/face_classification/trained_models/emotion_models/fer2013_emotion_training.log')
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

lines = file.plot.line(y='epoch', x='acc')
plt.title('CNN learning curves')
plt.xlabel('Accuracy')
plt.ylabel('Epoch')
# plt.legend(['training', 'validation'], loc='lower right')
plt.show()


import numpy as np

from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model('model/animal10.hdf5')

class_names = ['at', 'fil', 'inek', 'kedi', 'kelebek', 'kopek', 'koyun', 'orumcek', 'sincap', 'tavuk']

img = image.load_img('testler/merve.jpg', target_size= (224, 224))
# plt.imshow(img)
img.show()
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
result_b = model.predict(img)
print(class_names[np.argmax(result_b[0])])
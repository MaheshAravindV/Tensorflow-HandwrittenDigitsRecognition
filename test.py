import tensorflow,numpy as np
from PIL import Image,ImageOps
from tensorflow.keras.models import load_model
model = load_model('./model')
img = ImageOps.grayscale(Image.open('test.png'))
imgarr = np.array(img)/255
test = tensorflow.convert_to_tensor(imgarr, dtype=tensorflow.float64)
res = model.predict(tensorflow.reshape(test,[-1,28,28]))
l = [(res[0][i],i) for i in range(10)]
l.sort(reverse = True)
print(f'First {l[0]} Second {l[1]} Third {l[2]}')
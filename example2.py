# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#print(tf.__version__)

print('Preformatting image data...')

sub_size = 30
amount_of_big_images = 3

img_raw1 = mpimg.imread('data/scaled_down_300_300_0001.png')
img_aa1 = mpimg.imread('data/scaled_down_aa_300_300_0001.png')
img_raw2 = mpimg.imread('data/scaled_down_300_300_0002.png')
img_aa2 = mpimg.imread('data/scaled_down_aa_300_300_0002.png')
img_raw3 = mpimg.imread('data/scaled_down_300_300_0003.png')
img_aa3 = mpimg.imread('data/scaled_down_aa_300_300_0003.png')
img_raw4 = mpimg.imread('data/scaled_down_300_300_0003.png')
img_aa4 = mpimg.imread('data/scaled_down_aa_300_300_0004.png')





width = len(img_raw1)-sub_size
data_size_one_color = (len(img_raw1)-sub_size+1)*(len(img_raw1)-sub_size+1)
data_size = data_size_one_color * 3 * amount_of_big_images
#print(data_size)

sub_images_raw = np.zeros((data_size, sub_size, sub_size))
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*0,
                       :, :] = img_raw1[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*1,
                       :, :] = img_raw1[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*2,
                       :, :] = img_raw1[x:x+sub_size, y:y+sub_size, 2]

for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*3,
                       :, :] = img_raw2[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*4,
                       :, :] = img_raw2[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*5,
                       :, :] = img_raw2[x:x+sub_size, y:y+sub_size, 2]

for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*6,
                       :, :] = img_raw3[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*7,
                       :, :] = img_raw3[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*8,
                       :, :] = img_raw3[x:x+sub_size, y:y+sub_size, 2]





sub_images_aa = np.zeros((data_size, sub_size, sub_size))
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*0,
                      :, :] = img_aa1[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*1,
                      :, :] = img_aa1[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*2,
                      :, :] = img_aa1[x:x+sub_size, y:y+sub_size, 2]

for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*3,
                      :, :] = img_aa2[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*4,
                      :, :] = img_aa2[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*5,
                      :, :] = img_aa2[x:x+sub_size, y:y+sub_size, 2]

for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*6,
                      :, :] = img_aa3[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*7,
                      :, :] = img_aa3[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*8,
                      :, :] = img_aa3[x:x+sub_size, y:y+sub_size, 2]


print('Converted image data to smaller unicolor chunks')

plt.show()


sub_images_raw = sub_images_raw / 255.0
sub_images_aa = sub_images_aa / 255.0



model = keras.Sequential()
#model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(30, 30)))
model.add(keras.layers.Flatten(input_shape=(30,30)))
model.add(keras.layers.Dense(900, activation=keras.layers.LeakyReLU(alpha=0.3)))
model.add(keras.layers.Dense(900, activation=keras.layers.LeakyReLU(alpha=0.3)))
#model.add(keras.layers.Dense(900, activation='tanh'))
#model.add(keras.layers.Dense(900, activation='tanh'))
model.add(keras.layers.Dense(900))#, activation=keras.layers.LeakyReLU(alpha=0.3)))
#model.add(keras.layers.Dense(900, activation='tanh'))
model.add(keras.layers.Reshape((30, 30)))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(sub_images_raw, sub_images_aa, epochs=100)
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('Test accuracy:', test_acc)

#model.save('my_model.h5')

predictions = model.predict(sub_images_raw)


trigger = 0
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if(trigger == 0):
        plt.imshow(sub_images_raw[30*i], cmap=plt.cm.binary)
    if(trigger == 1):
        plt.imshow(predictions[30*(i-1)], cmap=plt.cm.binary)
    if(trigger == 2):
        plt.imshow(sub_images_aa[30*(i-2)], cmap=plt.cm.binary)
    trigger = trigger + 1
    if(trigger >= 3):
        trigger = 0
plt.savefig('plot.png')




trigger = 0
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    if(trigger == 0):
        plt.imshow(sub_images_raw[30*i], cmap=plt.cm.binary)
    if(trigger == 1):
        plt.imshow(predictions[30*(i-1)], cmap=plt.cm.binary)
    if(trigger == 2):
        plt.imshow(sub_images_aa[30*(i-2)], cmap=plt.cm.binary)
    trigger = trigger + 1
    if(trigger >= 3):
        trigger = 0
plt.show()


img_return = np.zeros(img_aa1.shape)
for x in range(0, width-1):
    for y in range(0, width-1):
         img_return[x:x+sub_size, y:y+sub_size, 0] = predictions[x*width+y+data_size_one_color*0,
                      :, :]*255
for x in range(0, width-1):
    for y in range(0, width-1):
        img_return[x:x+sub_size, y:y+sub_size, 1] =  predictions[x*width+y+data_size_one_color*1,
                      :, :]*255
for x in range(0, width-1):
    for y in range(0, width-1):
         img_return[x:x+sub_size, y:y+sub_size, 2] = predictions[x*width+y+data_size_one_color*2,
                      :, :]*255

plt.imshow(img_return)
plt.savefig('return_image.png')


exit()

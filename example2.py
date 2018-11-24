# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
#import cv2

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

print(tf.__version__)

img_raw = mpimg.imread('data/scaled_down_300_300_0001.png')
img_aa = mpimg.imread('data/scaled_down_aa_300_300_0001.png')


sub_size = 30

print(len(img_raw))
print(img_raw.shape)

width = len(img_raw)-sub_size
data_size_one_color = (len(img_raw)-sub_size+1)*(len(img_raw)-sub_size+1)
data_size = data_size_one_color * 3
print(data_size)

sub_images_raw = np.zeros((data_size, sub_size, sub_size))
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*0,
                       :, :] = img_raw[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*1,
                       :, :] = img_raw[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_raw[x*width+y+data_size_one_color*2,
                       :, :] = img_raw[x:x+sub_size, y:y+sub_size, 1]


sub_images_aa = np.zeros((data_size, sub_size, sub_size))
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*0,
                      :, :] = img_aa[x:x+sub_size, y:y+sub_size, 0]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*1,
                      :, :] = img_aa[x:x+sub_size, y:y+sub_size, 1]
for x in range(0, width-1):
    for y in range(0, width-1):
        sub_images_aa[x*width+y+data_size_one_color*2,
                      :, :] = img_aa[x:x+sub_size, y:y+sub_size, 1]


# print(sub_images_raw.shape)
print('Converted image data to smaller unicolor chunks')


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()
print(train_labels.shape)

#trigger = True
#plt.figure(figsize=(10, 10))
#for i in range(36):
#    plt.subplot(6, 6, i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    if(trigger):
#        plt.imshow(sub_images_raw[30*i], cmap=plt.cm.binary)
#    if(not trigger):
#        plt.imshow(sub_images_aa[30*(i-1)], cmap=plt.cm.binary)
#    trigger = not trigger

# plt.imshow(train_images[0])
# plt.imshow(sub_images_aa[200000])
plt.show()


sub_images_raw = sub_images_raw / 255.0
sub_images_aa = sub_images_aa / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30, 30)),
    keras.layers.Dense(900, activation=keras.layers.ELU(alpha=1.0)),
    keras.layers.Dense(900, activation=keras.layers.ELU(alpha=1.0)),
    keras.layers.Dense(900, activation=keras.layers.LeakyReLU(alpha=0.3)),
    keras.layers.Reshape((30, 30))
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(sub_images_raw, sub_images_aa, epochs=10)
#test_loss, test_acc = model.evaluate(test_images, test_labels)

#print('Test accuracy:', test_acc)

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




exit()

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#load dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape, train_labels.shape) #((60000, 28, 28), (60000,))
print(test_images.shape, test_labels.shape) #((10000, 28, 28), (10000,))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images[0]) #2D image array

#To display image depending on index
"""our images are gray-scale each pixel will have a value between 0–255. 0 means WHITE. 255 means BLACK"""

# plt.figure()
# plt.imshow(train_images[0], cmap='binary')
# plt.colorbar()
# plt.xticks([])
# plt.yticks([])
# plt.show()

#Preprocess the Data
"""reduce the pixel value from 0–255 to 0–1."""
train_images = train_images/255.0
test_images = test_images/255.0

plt.figure()
plt.imshow(train_images[0], cmap='binary')
plt.colorbar()
plt.xticks([])
plt.yticks([])
plt.show()

#Let us look at the first 25 images
plt.figure(figsize=(20, 10))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(train_images[i], cmap='binary')
  plt.xlabel(class_names[train_labels[i]])
plt.show()

#Building Model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy",metrics=['accuracy'])

"""We created a neural network with 3 layers. The first layer is the input layer which is all the pixels in our image.
The second layer is a hidden layer with 128 neurons(it is just an arbitrary value) this is where the computation takes 
place (here is where the weights and the biases are found). The third layer is the output layer with 10 neurons
(because we have 10 different categories)"""

"""When compiling the model, we need 2 important things a loss function and an optimizer.
We need a loss function to know how far our prediction is from the original value. We need an optimizer to reduce the
distance between the prediction and the original value"""

#Train the Model
model.fit(train_images, train_labels, epochs=10)

#Model Evaluation
acc = model.evaluate(test_images, test_labels)
print(acc) #88%

#Predict
classification = model.predict(test_images)

print(classification[0]) #gives classification array
print(np.argmax(classification[0]), test_labels[0]) #prints the max value prediction
print(class_names[np.argmax(classification[0])])

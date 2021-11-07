import tensorflow as tf
from network import classifier
from utils import read_config


(train_images, train_labels), (test_images, test_labels)  = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0

test_images = test_images / 255.0

config = read_config("./model-config.yaml")
num_classes = config["train"]["num_classes"]
learning_rate = config["train"]["lr"]
batch_size = config["train"]["batch_size"]
epochs = config["train"]["epochs"]
checkpoint_filepath = config["train"]["save_path"]

model = classifier(num_classes)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
model.compile(
    optimizer = optimizer ,
    loss= loss_fn,
    metrics=['accuracy'])

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filepath,save_weights_only=True,verbose=1)
model.fit(train_images,train_labels,epochs=epochs,callbacks=[checkpoint_callback])

model.save(checkpoint_filepath+"/final_model.h5")
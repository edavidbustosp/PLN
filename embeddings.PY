#IMPORTAR LIBRERÍAS

import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

#3. Cargue el conjunto de datos MNIST

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

rng = np.random.default_rng(42)
idx = rng.choice(len(x_test), size=1000, replace=False)
x_sample = x_test[idx]
y_sample = y_test[idx]

#4. Construya y entrene una red neuronal simple para clasificar los dígitos.

model = tf.keras.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu", name="embedding_layer"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=256, validation_split=0.1)

#5. Cree un "modelo extractor" que entregue la salida de embedding_layer

extractor = tf.keras.Model(inputs=model.inputs, outputs=model.get_layer("embedding_layer").output)
embeddings = extractor.predict(x_sample)
print(embeddings.shape)   # (1000, 128)

#Cree una carpeta de logs y guarde el archivo metadata.tsv 

log_dir = "logs/mnist-projector"
os.makedirs(log_dir, exist_ok=True)

with open(os.path.join(log_dir, "metadata.tsv"), "w") as f:
    for label in y_sample:
        f.write(f"{label}\n")

#2. Guarde los embeddings como una variable de TensorFlow

weights = tf.Variable(embeddings, name="embedding")
checkpoint = tf.train.Checkpoint(embedding=weights)
checkpoint.save(os.path.join(log_dir, "embedding.ckpt"))

#3. Configure el Projector,

config = projector.ProjectorConfig()
embedding_config = config.embeddings.add()
embedding_config.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
embedding_config.metadata_path = "metadata.tsv"
projector.visualize_embeddings(log_dir, config)

#

#%load_ext tensorboard
#%tensorboard --logdir logs/mnist-projector


#

import math
from PIL import Image

grid = math.ceil(math.sqrt(len(x_sample)))
sprite_dim = 28
sprite_image = Image.new("L", (grid * sprite_dim, grid * sprite_dim), color=0)
for i, img in enumerate(x_sample):
    row, col = divmod(i, grid)
    im = Image.fromarray((img * 255).astype(np.uint8))
    sprite_image.paste(im, (col * sprite_dim, row * sprite_dim))
sprite_image.save(os.path.join(log_dir, "sprite.png"))


#

embedding_config.sprite.image_path = "sprite.png"
embedding_config.sprite.single_image_dim.extend([sprite_dim, sprite_dim])
projector.visualize_embeddings(log_dir, config)
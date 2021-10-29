from inspect import EndOfBlock
import os
import time
import logging

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.ops.gen_array_ops import shape

from text_cnn import TextCNN
from data_processing import DataLoader


def train_old():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f'./logs/model_train.log', mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    epochs = 200
    log_step = 10
    batch_size = 2000
    learning_rate = 1e-3

    # load data
    dataloader = DataLoader()

    train_x, train_y = dataloader.get_data(mode='train')
    train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)

    # init model
    model = TextCNN()
    model._set_inputs(tf.TensorSpec(shape=(batch_size, dataloader.max_len)))

    # init optim
    optimizer = keras.optimizers.Adam(learning_rate)

    # begin train
    for epoch in range(epochs):
        loss_list = []
        for step, (x, true_y) in enumerate(train_dataset):
            
            with tf.GradientTape() as tape:
                pred_y = model(x)
                loss = keras.losses.categorical_crossentropy(true_y, pred_y)

            grad = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            
            loss_list.append(tf.reduce_mean(loss).numpy())
            if step % log_step == 0:
                logging.info(f'No.{epoch}-> step.{step} loss: {tf.reduce_mean(loss).numpy()}')
                # print(f'No.{epoch}--> step.{step} training loss: {tf.reduce_mean(loss).numpy()}')

        logging.info(f'No.{epoch}----------------> training loss: {tf.reduce_mean(loss_list).numpy()}')
        print(f'No.{epoch}-------------> training loss: {tf.reduce_mean(loss_list).numpy()}')

    return model



def train():
    epochs = 200
    batch_size = 2000
    learning_rate = 1e-3

    # load data
    dataloader = DataLoader()

    train_x, train_y = dataloader.get_data(mode='train')
    train_x_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size)


    # init model
    model = TextCNN()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.CategoricalCrossentropy()
    )

    tf.config.experimental_run_functions_eagerly(True)
    model.fit(x=train_x_dataset, batch_size=batch_size, epochs=epochs)

    return model


if __name__ == "__main__":

    # physical_devices = tf.config.list_physical_devices('CPU')
    # tf.config.set_visible_devices(physical_devices)

    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = train()

    # save model
    save_path = './models/'

    model.save_weights(os.path.join(save_path, 'model_weights.h5'))
    # model.save(os.path.join(save_path, 'local_model'), save_format='tf')

    # tf.saved_model.save(model, os.path.join(save_path, 'remote_model'))
    

    

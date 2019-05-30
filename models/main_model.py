from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten, Concatenate, Input, MaxPooling2D
from keras.models import Model

INPUT_SHAPE, RADAR_SHAPE = (66, 200, 3), (20, 20, 1)


def main_model():
    # image model
    img_input = Input(shape=INPUT_SHAPE)
    img_model = (Lambda(lambda x: x / 127.5 - 1.0, input_shape=INPUT_SHAPE))(img_input)
    img_model = (Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Conv2D(64, (3, 3), activation='elu'))(img_model)
    img_model = (Flatten())(img_model)
    img_model = (Dense(100, activation='elu'))(img_model)

    # radar model
    radar_input = Input(shape=RADAR_SHAPE)
    radar_model = (Conv2D(32, (5, 5), activation='elu'))(radar_input)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Conv2D(64, (5, 5), activation='elu'))(radar_model)
    radar_model = (MaxPooling2D((2, 2), strides=(2, 2)))(radar_model)
    radar_model = (Flatten())(radar_model)
    radar_model = (Dense(10, activation='elu'))(radar_model)

    # speed
    speed_input = Input(shape=(1,))

    # combined model
    out = Concatenate()([img_model, radar_model])
    out = (Dense(50, activation='elu'))(out)
    out = Concatenate()([out, speed_input])
    out = (Dense(10, activation='elu'))(out)
    out = (Dense(1))(out)

    model = Model(inputs=[img_input, radar_input, speed_input], outputs=out)

    return model

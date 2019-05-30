from train.data_loader import load_data
from keras.callbacks import TensorBoard
from models import main_model as model

print("Loading Model ...")

model = model.main_model()
model.compile(optimizer='Adam', loss='mean_squared_error', metrics=['accuracy'])

print("Starting Training...")

batch_count = 0

try:
    for i in range(0, 50):  # train 50 epochs
        print('----------- On Epoch: ' + str(i) + ' ----------')
        for x_train, y_train, x_test, y_test in load_data():

            model.fit(x_train, y_train, verbose=1, epochs=1, validation_data=(x_test, y_test), callbacks=[TensorBoard(log_dir="./log_dir")])

            batch_count += 1

            if (batch_count % 20) == 0:
                print('Saving checkpoint ' + str(batch_count))
                model.save('../ckep/model_checkpoint' + str(batch_count) + '.h5')
                print('Checkpoint saved.')

except Exception as e:
    print(str(e))
    model.save('main_model.h5')
    print('Model saved.')

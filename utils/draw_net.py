from keras.utils import plot_model
from models.main_model import main_model

plot_model(main_model(), to_file='model.png', show_shapes=True)   # 使用。。。画神经网络

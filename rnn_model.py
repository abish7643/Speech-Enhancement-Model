import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, test
from tensorflow.keras.layers import Input, Dense, GRU, concatenate, Dropout, Flatten, Activation
from tensorflow.keras.models import Model

train_model = True
model_dir = "Prototyping/Trained Model"
trained_model = "model.h5"

gen_dataset_dir = "Generated Features"
gen_dataset = "feature_dataset_ms_psf_.npz"


def check_gpu():
    if test.gpu_device_name():
        print('Default GPU Device: {}'.format(test.gpu_device_name()))
        return True
    else:
        print("GPU Not Found")
        return False


def model_gru(regulazier=0.001, input_size=52, output_size=22):
    _input = Input(shape=(None, input_size))
    _x = GRU(96, return_sequences=True,
             )(_input)
    _x = Dropout(0.3)(_x)
    _x = GRU(96, return_sequences=True,
             )(_x)
    _x = Dropout(0.3)(_x)
    _x = GRU(48, return_sequences=True,
             )(_x)
    _x = Dropout(0.3)(_x)
    _x = Dense(output_size)(_x)
    _output = Activation("hard_sigmoid")(_x)
    _model = Model(inputs=_input, outputs=[_output])

    return _model


def plot_analytics(history, axis=None, label=None, legend="upper right"):
    if label is None:
        label = ["", ""]

    plt.style.use(['fivethirtyeight'])
    plt.rc('font', size=20)
    plt.rc('axes', edgecolor='grey')
    plt.rc('axes', labelsize='32')
    plt.rc('legend', fontsize=24)

    # Plot Figure
    plt.figure(figsize=(18, 5))
    plt.plot(history.history[axis[0]], label=label[0])
    plt.plot(history.history[axis[1]], label=label[1])
    plt.ylabel(axis[0].capitalize())
    plt.xlabel("Epochs")
    plt.yticks(size=22)
    plt.xticks(size=22)
    plt.legend(loc=legend)
    # plt.title(axis[0].capitalize(), fontsize=50)
    plt.show()


def load_generated_dataset(file_dir, features_per_frame=None, window_size=None):

    if features_per_frame is None:
        features_per_frame = [42, 22]

    # Load File
    print("\nLoading Dataset: {}".format(file_dir.split("/")[-1]))

    with np.load(file_dir) as data:

        # Load Features from File
        _speech_features = data["speech_features"]
        _gains = data["gains"]

        _speech_features = _speech_features[:features_per_frame[0]]
        _gains = _gains[:features_per_frame[1]]

        # Clip Gains Outside 0 and 1
        _gains = np.clip(_gains, 0, 1)

        # Take Transpose
        _speech_features = _speech_features.transpose()
        _gains = _gains.transpose()

        if window_size is not None:
            # Reshape Arrays to (Window Size, Number of Sequences, Features Per Frame)
            _number_of_sequences = int(len(_speech_features) / window_size)

            _speech_features = _speech_features[:_number_of_sequences * window_size]
            _gains = _gains[:_number_of_sequences * window_size]

            _speech_features = np.reshape(_speech_features, (window_size, _number_of_sequences, features_per_frame[0]))
            _gains = np.reshape(_gains, (window_size, _number_of_sequences, features_per_frame[1]))

        print("Loaded Dataset: {} & {}".format(_speech_features.shape, _gains.shape))

        return _speech_features, _gains


if __name__ == "__main__":

    if train_model and check_gpu():
        print("\nTrain Model\n")

        # Load Dataset
        x, y = load_generated_dataset(file_dir="/".join([gen_dataset_dir, gen_dataset]),
                                      window_size=2048, features_per_frame=[40, 22])

        # Input Features
        _model_gru = model_gru(input_size=40, regulazier=0.001)
        _optimizer = keras.optimizers.Adam(learning_rate=0.001)
        _model_gru.compile(optimizer=_optimizer,
                           loss="mse", metrics=["accuracy"], )

        # Model Summary & Train Model
        _model_gru.summary()

        print("Training Model")
        _history = _model_gru.fit(x, y,
                                  validation_split=0.2,
                                  batch_size=32, epochs=120,
                                  shuffle=False)
        print("\nTraining Complete\n")

        # Plot
        plot_analytics(_history, axis=["loss", "val_loss"],
                       label=["Train", "Test"])
        plot_analytics(_history, axis=["accuracy", "val_accuracy"],
                       label=["Train", "Test"], legend="lower right")

        # Clear Memory
        x, y = 0, 0

        # Save Model
        print("\nSaving Model to {} as {}".format(model_dir, trained_model))
        _model_gru.save(filepath="/".join([model_dir, trained_model]),
                        save_format="h5")
        print("Saved Model {}\n".format(trained_model))

    print("\nProcess Completed")

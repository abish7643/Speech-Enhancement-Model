import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras, test
from tensorflow.keras.layers import Input, Dense, GRU, concatenate, Dropout, Flatten, Activation
from tensorflow.keras.models import Model

train_model = True
model_dir = "Prototyping/Trained Model"
trained_model = "model_vad.h5"

gen_dataset_dir = "Generated Features"
gen_dataset = "feature_dataset_ms_librosa_vad.npz"


def check_gpu():
    if test.gpu_device_name():
        print('Default GPU Device: {}'.format(test.gpu_device_name()))
        return True
    else:
        print("GPU Not Found")
        return False


def model_gru(regulazier=0.001, input_size=52, output_size=22):
    _input = Input(shape=(None, input_size))
    _x = GRU(96, return_sequences=True, )(_input)
    _x = Dropout(0.3)(_x)
    _x = GRU(96, return_sequences=True, )(_x)
    _x = Dropout(0.3)(_x)
    _x = GRU(48, return_sequences=True, )(_x)
    _x = Dropout(0.3)(_x)
    _x = Dense(output_size)(_x)
    _output = Activation("hard_sigmoid")(_x)

    _model = Model(inputs=_input, outputs=[_output])

    return _model


def model_gru_vad(regulazier=0.001, input_size=52, output_size=22):
    _input = Input(shape=(None, input_size))

    # VAD
    _x_vad_1 = GRU(48, return_sequences=True, )(_input)
    _x_vad_1 = Dropout(0.3)(_x_vad_1)
    _x_vad_2 = GRU(24, return_sequences=True, )(_x_vad_1)
    _x_vad_2 = Dropout(0.3)(_x_vad_2)
    _x_vad = Dense(1)(_x_vad_2)
    _vad_output = Activation("hard_sigmoid")(_x_vad)

    # Band Gains
    _x_gains = GRU(64, return_sequences=True, )(_input)
    _x_gains = concatenate([_x_gains, _x_vad_1, _x_vad_2])
    _x_gains = GRU(96, return_sequences=True, )(_x_gains)
    _x_gains = Dropout(0.3)(_x_gains)
    _x_gains = GRU(48, return_sequences=True, )(_x_gains)
    _x_gains = Dropout(0.3)(_x_gains)
    _x_gains = Dense(output_size)(_x_gains)
    _gains_output = Activation("hard_sigmoid")(_x_gains)

    _model = Model(inputs=_input, outputs=[_gains_output, _vad_output])

    return _model


def plot_analytics(history, axis=None, label=None, title="", legend="upper right", save=True):
    if label is None:
        label = ["", ""]

    # plt.style.use(['fivethirtyeight'])
    plt.rc('font', size=20)
    # plt.rc('axes', edgecolor='grey')
    # plt.rc('axes', labelsize='32')
    plt.rc('legend', fontsize=24)

    # Plot Figure
    plt.figure(figsize=(18, 5))
    plt.plot(history.history[axis[0]], label=label[0])
    plt.plot(history.history[axis[1]], label=label[1])
    plt.xlabel("Epochs")
    plt.ylabel(axis[0].capitalize())
    plt.yticks(size=22)
    plt.xticks(size=22)
    plt.legend(loc=legend)
    plt.title(title)
    if save:
        plt.savefig("".join(["../", title, ".png"]), bbox_inches='tight')
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
        _vad = data["vad"]

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
            _vad = _vad[:_number_of_sequences * window_size]

            _speech_features = np.reshape(_speech_features, (window_size, _number_of_sequences, features_per_frame[0]))
            _gains = np.reshape(_gains, (window_size, _number_of_sequences, features_per_frame[1]))
            _vad = np.reshape(_vad, (window_size, _number_of_sequences, 1))

        print("Loaded Dataset: {} & {} & {}".format(_speech_features.shape, _gains.shape, _vad.shape))

        return _speech_features, _gains, _vad


if __name__ == "__main__":

    if train_model and check_gpu():
        print("\nTrain Model\n")

        # Load Dataset
        x, y, y_vad = load_generated_dataset(file_dir="/".join([gen_dataset_dir, gen_dataset]),
                                             window_size=2048, features_per_frame=[40, 22])

        # Input Features
        _model_gru = model_gru_vad(input_size=40, regulazier=0.001)
        _optimizer = keras.optimizers.Adam(learning_rate=0.001)
        _model_gru.compile(optimizer=_optimizer,
                           loss="mse", metrics=["accuracy"], )

        # Model Summary & Train Model
        _model_gru.summary()

        print("Training Model")
        _history = _model_gru.fit(x, [y, y_vad],
                                  validation_split=0.2,
                                  batch_size=32, epochs=120,
                                  shuffle=False)
        print("\nTraining Complete\n")

        # Clear Memory
        x, y, y_vad = 0, 0, 0

        # Save Model
        print("\nSaving Model to {} as {}".format(model_dir, trained_model))
        _model_gru.save(filepath="/".join([model_dir, trained_model]),
                        save_format="h5")
        print("Saved Model {}\n".format(trained_model))

        # Plot
        plot_analytics(_history, axis=["loss", "val_loss"],
                       label=["Train", "Test"], title="Loss")
        # plot_analytics(_history, axis=["accuracy", "val_accuracy"],
        #                label=["Train", "Test"], legend="lower right", title="Accuracy")

        plot_analytics(_history, axis=["activation_loss", "val_activation_loss"],
                       label=["Train", "Test"], title="Loss (VAD)")
        plot_analytics(_history, axis=["activation_1_loss", "val_activation_1_loss"],
                       label=["Train", "Test"], legend="lower right", title="Loss (Gains)")

        plot_analytics(_history, axis=["activation_accuracy", "val_activation_accuracy"],
                       label=["Train", "Test"], legend="lower right", title="Accuracy (VAD)")
        plot_analytics(_history, axis=["activation_1_accuracy", "val_activation_1_accuracy"],
                       label=["Train", "Test"], legend="lower right", title="Accuracy (Gains)")

    print("\nProcess Completed")

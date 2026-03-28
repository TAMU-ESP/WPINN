import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from models.cnn_transformer_nn import CNNTransformerNet
from keras import backend as K
import itertools

class ConvModel:

    def __init__(self, x_flow, x_time, x_beat, y_BP,
                 train_ind, val_ind, test_ind):
        # Train data
        self.x_flow_train = x_flow[train_ind]
        self.x_time_train = x_time[train_ind]
        self.x_beat_train = x_beat[train_ind]
        self.y_BP_train = y_BP[train_ind]
        # Val data
        self.x_flow_val = x_flow[val_ind]
        self.x_time_val = x_time[val_ind]
        self.x_beat_val = x_beat[val_ind]
        self.y_BP_val = y_BP[val_ind]
        # Test data
        self.x_flow_test = x_flow[test_ind]
        self.x_time_test = x_time[test_ind]
        self.x_beat_test = x_beat[test_ind]
        self.y_BP_test = y_BP[test_ind]

    def preprocess(self):
        # Waveforms means & stds
            m_flow, s_flow = np.mean(self.x_flow_train), np.std(self.x_flow_train)
            m_time, s_time = np.mean(self.x_time_train), np.std(self.x_time_train)
            m_beat, s_beat = np.mean(self.x_beat_train), np.std(self.x_beat_train)
            m_BP, s_BP = np.mean(self.y_BP_train), np.std(self.y_BP_train)

            x_flow_train, x_flow_val, x_flow_test = ((self.x_flow_train - m_flow) / s_flow,
                                                     (self.x_flow_val - m_flow) / s_flow,
                                                     (self.x_flow_test - m_flow) / s_flow)
            x_time_train, x_time_val, x_time_test = ((self.x_time_train - m_time) / s_time,
                                                     (self.x_time_val - m_time) / s_time,
                                                     (self.x_time_test - m_time) / s_time)
            x_beat_train, x_beat_val, x_beat_test = ((self.x_beat_train - m_beat) / s_beat,
                                                     (self.x_beat_val - m_beat) / s_beat,
                                                     (self.x_beat_test - m_beat) / s_beat)
            y_BP_train, y_BP_val, y_BP_test = ((self.y_BP_train - m_BP) / s_BP,
                                               (self.y_BP_val - m_BP) / s_BP,
                                               (self.y_BP_test - m_BP) / s_BP)

            x_wave_train, y_train = np.dstack([x_flow_train, x_time_train, x_beat_train]), np.dstack([y_BP_train])
            x_wave_val, y_val = np.dstack([x_flow_val, x_time_val, x_beat_val]), np.dstack([y_BP_val])
            x_wave_test, y_test = np.dstack([x_flow_test, x_time_test, x_beat_test]), np.dstack([y_BP_test])

            x_wave_val_test = np.concatenate([x_wave_val, x_wave_test], axis=0)
            return (m_flow, s_flow, m_time, s_time, m_BP, s_BP,
                    x_wave_train , y_train,
                    x_wave_val , y_val,
                    x_wave_test, y_test,
                    x_wave_val_test)

    def model_train(self, batch=16, epochs=64):
        # Import arrays, models, etc.
        (m_flow, s_flow, m_time, s_time, m_BP, s_BP,
         x_wave_train, y_train,
         x_wave_val, y_val,
         x_wave_test, y_test,
         x_wave_val_test) = ConvModel.preprocess(self)
        windk_model, _, unc_param = CNNTransformerNet(parameter_type=None).nn_model()
        opt = tf.keras.optimizers.Adam()

        # Constants
        thr = np.inf

        # Loss curve lists
        conv_loss = []
        val_loss = []

        # Iterate for specified number of epochs
        for e in range(epochs):
            # Get supervised train indexes for batch
            np.random.shuffle(np.arange(len(x_wave_train)))
            train_batches = np.array_split(np.arange(len(x_wave_train)), np.ceil(len(x_wave_train) / batch))
            # Get val & test indexes for batch
            val_test_idx = np.arange(len(x_wave_val) + len(x_wave_test))
            np.random.shuffle(val_test_idx)
            val_test_batches = np.array_split(val_test_idx, np.ceil((len(x_wave_val) + len(x_wave_test)) / batch))
            val_test_batches = list(itertools.islice(itertools.cycle(val_test_batches), len(train_batches)))
            # Iterate through all train batches (val+test batches very large)
            for b in range(len(train_batches)):
                wave_tensor = tf.convert_to_tensor(x_wave_train[train_batches[b]], dtype=tf.float32)
                ref_BP_tensor = tf.convert_to_tensor(y_train[train_batches[b]], dtype=tf.float32)

                # Calculate gradient
                with (tf.GradientTape() as tape):
                    yp_BP_o = windk_model(wave_tensor, training=True)

                    # Define BP & parameter outputs
                    yp_BP_d = yp_BP_o[:, :, 0]

                    # Conventional MSE loss
                    loss_mse_BP = K.mean(K.square(ref_BP_tensor[:, :, 0] - yp_BP_d))

                    # Total loss
                    unc_param = tf.cast(unc_param, tf.float64)
                    # Calculate weights
                    e_d = tf.abs(unc_param[0])
                    s_d = tf.math.log(tf.square(e_d))
                    w_d = (0.5 * tf.math.exp(-s_d)) * 0
                    loss_total = (1 + w_d) * tf.cast(loss_mse_BP, tf.float64) + (s_d * 0)

                # Apply gradients & opt
                gradients = tape.gradient(loss_total, windk_model.trainable_weights + [unc_param])
                opt.apply_gradients(zip(gradients, windk_model.trainable_weights + [unc_param]))

            # Append to loss curves
            conv_loss.append(loss_mse_BP)
            val_pred = windk_model.predict(x_wave_val, verbose=0)[:, :, 0]
            val_ref = y_val[:, :, 0]
            val_mse = mean_squared_error(val_pred.flatten(), val_ref.flatten())
            val_loss.append(val_mse)

            # Save the best model according to validation MSE
            if val_mse < thr:
                best_model = tf.keras.models.clone_model(windk_model)
                best_model.build(windk_model.input_shape)
                best_model.set_weights(windk_model.get_weights())
                thr = val_mse

        # Final Reference & Predicted Test BP Data
        y_final_test = y_test[:, :, 0] * s_BP + m_BP
        yp_final_test = best_model.predict(x_wave_test, verbose=0)[:, :, 0] * s_BP + m_BP
        return (best_model, y_final_test, yp_final_test,
                np.array(conv_loss), np.array(val_loss))

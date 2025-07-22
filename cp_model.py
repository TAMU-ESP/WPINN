import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from cnn_transformer_nn import CNNTransformerNet
from keras import backend as K
import itertools
tf.random.set_seed(0)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class CPModel:

    def __init__(self, x_flow_train, x_time_train, x_beat_train, y_BP_train,
               x_flow_val, x_time_val, x_beat_val, y_BP_val,
               x_flow_test, x_time_test, x_beat_test, y_BP_test):
        self.x_flow_train = x_flow_train
        self.x_time_train = x_time_train
        self.x_beat_train = x_beat_train
        self.y_BP_train = y_BP_train
        self.x_flow_val = x_flow_val
        self.x_time_val = x_time_val
        self.x_beat_val = x_beat_val
        self.y_BP_val = y_BP_val
        self.x_flow_test = x_flow_test
        self.x_time_test = x_time_test
        self.x_beat_test = x_beat_test
        self.y_BP_test = y_BP_test

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

    def model_train(self, Ew_model=2, physics_weight=1, batch=16, epochs=64):
        # Import arrays, models, etc.
        (m_flow, s_flow, m_time, s_time, m_BP, s_BP,
         x_wave_train, y_train,
         x_wave_val, y_val,
         x_wave_test, y_test,
         x_wave_val_test) = CPModel.preprocess(self)
        windk_model, params, unc_param = CNNTransformerNet.nn_model(input_shape=(64, 3), parameter_type='cp')
        optimizer_windk_model = tf.keras.optimizers.Adam(clipnorm=10)

        # Constants
        pi, epsilon, thr = tf.constant(np.pi), tf.constant(1e-6), np.inf

        # Loss curve lists
        conv_loss, phys_loss = [], []
        val_loss, test_loss = [], []

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
                wave_tensor = tf.convert_to_tensor(np.concatenate([x_wave_train[train_batches[b]],
                                                                   x_wave_val_test[val_test_batches[b]]], axis=0),
                                                                   dtype=tf.float32)
                ref_BP_tensor = tf.convert_to_tensor(y_train[train_batches[b]], dtype=tf.float32)
                ref_BP_tensor = ref_BP_tensor * s_BP + m_BP

                # Calculate gradient
                with (tf.GradientTape() as tape):
                    # Compute the first derivative of dy/dt
                    with tf.GradientTape() as deriv1:
                        deriv1.watch(wave_tensor)
                        yp_BP_o = windk_model(wave_tensor, training=True)
                    dyp_BP_dt = deriv1.gradient(yp_BP_o, wave_tensor)[:, :, 1]
                    dyp_BP_dt = dyp_BP_dt * (s_BP / s_time)

                    # Define BP & parameter outputs
                    yp_BP_d = yp_BP_o[:, :, 0]  # reshape
                    yp_BP_d = yp_BP_d * s_BP + m_BP

                    # Calc dI/dt
                    flow = wave_tensor[:, :, 0] * s_flow + m_flow
                    time = wave_tensor[:, :, 1] * s_time + m_time
                    dI_flow_dt = tf.convert_to_tensor(np.gradient(flow, axis=-1) / np.gradient(time, axis=-1),
                                                      dtype=tf.float32)

                    # Params
                    Z = 0.1*tf.tanh(params[0]) + (0.1 + 1e-2)
                    C = 2.5*tf.tanh(params[1]) + (2.5 + 1e-2)
                    R = tf.clip_by_value(tf.abs(params[2]), 1e-2, 100)

                    # Defne individual losses #

                    # Physics loss
                    if Ew_model == 3:
                        physics = ((dyp_BP_dt / (Z * dI_flow_dt + epsilon)) +
                                   (yp_BP_d / (C * Z * R * dI_flow_dt + epsilon)) -
                                   (((1 / Z) + (1 / R)) * (flow / (C * dI_flow_dt + epsilon))) - 1)
                    else:
                        physics = dyp_BP_dt + (yp_BP_d / (C * R + epsilon)) - (flow / C)
                    loss_physics = K.mean(K.square(physics / s_flow))

                    # Conventional MSE loss
                    loss_mse_BP = K.mean(K.square((ref_BP_tensor[:, :, 0] - yp_BP_d[:len(ref_BP_tensor)]) / s_BP))

                    # Total loss
                    unc_param = tf.cast(unc_param, tf.float64)
                    # Calculate weights
                    e_d = tf.abs(unc_param[0])
                    s_d = tf.math.log(tf.square(e_d))
                    e_p = tf.abs(unc_param[1])
                    s_p = tf.math.log(tf.square(e_p))
                    w_d = 0.5 * tf.math.exp(-s_d)
                    w_p = 0.5 * tf.math.exp(-s_d)
                    # Ensure loss_mse_BP and loss_physics are float64
                    loss_mse_BP = w_d * tf.cast(loss_mse_BP, tf.float64)
                    loss_physics = w_p * tf.cast(loss_physics, tf.float64)
                    c1 = s_d + s_p
                    c2 = s_d + s_p*0
                    # Calculate total loss
                    loss_total = loss_mse_BP + loss_physics + c1
                    if physics_weight == 0:
                        loss_total =  loss_mse_BP + 0 * loss_physics + c2

                # Apply gradients & opt
                gradients = tape.gradient(loss_total, windk_model.trainable_weights + [params, unc_param])
                optimizer_windk_model.apply_gradients(zip(gradients,
                                                          windk_model.trainable_weights + [params, unc_param]))

            # Append to loss curves
            conv_loss.append(loss_mse_BP), phys_loss.append(loss_physics)
            val_pred = windk_model.predict(x_wave_val)[:, :, 0]
            val_ref = y_val[:, :, 0]
            test_pred = windk_model.predict(x_wave_test)[:, :, 0]
            test_ref = y_test[:, :, 0]
            val_mse = mean_squared_error(val_pred.flatten(), val_ref.flatten())
            test_mse = mean_squared_error(test_pred.flatten(), test_ref.flatten())
            val_loss.append(val_mse), test_loss.append(test_mse)

            # Save the best model according to validation MSE
            if val_mse < thr:
                best_model = tf.keras.models.clone_model(windk_model)
                best_model.build(windk_model.input_shape)
                best_model.set_weights(windk_model.get_weights())
                thr = val_mse

        # Final Reference & Predicted Test BP Data
        y_final_test = y_test[:, :, 0] * s_BP + m_BP
        yp_final_test = best_model.predict(x_wave_test)[:, :, 0] * s_BP + m_BP

        return (best_model, y_final_test, yp_final_test,
                np.array(conv_loss), np.array(phys_loss), np.array(val_loss), np.array(test_loss))
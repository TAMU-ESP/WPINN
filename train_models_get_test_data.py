import pandas as pd
import tensorflow as tf
from conv_model import ConvModel
from cp_model import CPModel
from bbp_model import BBPModel
from pdp_model import PDPModel
########################################################################################################################
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
########################################################################################################################
df = pd.concat([pd.read_pickle('ring_cpt.pkl'), pd.read_pickle('hypertensive_hgvcpt.pkl')])
epochs = 64
########################################################################################################################
for bp_type in ['SBP', 'DBP']:
    for perc in [5.0, 2.5]:
        batch = int((perc / 5) * 16)
        df_split = df[(df['bp_type'] == bp_type) & (df['percentile'] == perc)]
        for model in ['conv', 'cp', 'bbp', 'pdp']:
            for Ew_model in [2, 3]:
                if model == 'conv' and Ew_model == 3:
                    continue
                y_true, y_pred = [], []
                for s in df_split['subject'].unique():
                    df_s = df_split[df_split['subject'] == s]
                    if model == 'conv':
                        (_, y_final_test, yp_final_test,
                         _, _) = (
                        ConvModel(df_s['x_flow_train'].to_numpy()[0], df_s['x_time_train'].to_numpy()[0], df_s['x_beat_train'].to_numpy()[0], df_s['y_BP_train'].to_numpy()[0],
                        df_s['x_flow_val'].to_numpy()[0], df_s['x_time_val'].to_numpy()[0], df_s['x_beat_val'].to_numpy()[0], df_s['y_BP_val'].to_numpy()[0],
                        df_s['x_flow_test'].to_numpy()[0], df_s['x_time_test'].to_numpy()[0], df_s['x_beat_test'].to_numpy()[0], df_s['y_BP_test'].to_numpy()[0]).model_train(batch=batch, epochs=epochs))
                    elif model == 'cp':
                        (_, y_final_test, yp_final_test,
                         _, _, _) = (
                        CPModel(df_s['x_flow_train'].to_numpy()[0], df_s['x_time_train'].to_numpy()[0], df_s['x_beat_train'].to_numpy()[0], df_s['y_BP_train'].to_numpy()[0],
                        df_s['x_flow_val'].to_numpy()[0], df_s['x_time_val'].to_numpy()[0], df_s['x_beat_val'].to_numpy()[0], df_s['y_BP_val'].to_numpy()[0],
                        df_s['x_flow_test'].to_numpy()[0], df_s['x_time_test'].to_numpy()[0], df_s['x_beat_test'].to_numpy()[0], df_s['y_BP_test'].to_numpy()[0]).model_train(Ew_model=Ew_model, batch=batch, epochs=epochs))
                    elif model == 'bbp':
                        (_, y_final_test, yp_final_test,
                         _, _, _) = (
                        BBPModel(df_s['x_flow_train'].to_numpy()[0], df_s['x_time_train'].to_numpy()[0], df_s['x_beat_train'].to_numpy()[0], df_s['x_SV_train'].to_numpy()[0], df_s['y_BP_train'].to_numpy()[0],
                        df_s['x_flow_val'].to_numpy()[0], df_s['x_time_val'].to_numpy()[0], df_s['x_beat_val'].to_numpy()[0], df_s['x_SV_val'].to_numpy()[0], df_s['y_BP_val'].to_numpy()[0],
                        df_s['x_flow_test'].to_numpy()[0], df_s['x_time_test'].to_numpy()[0], df_s['x_beat_test'].to_numpy()[0], df_s['x_SV_test'].to_numpy()[0], df_s['y_BP_test'].to_numpy()[0]).model_train(Ew_model=Ew_model, batch=batch, epochs=epochs))
                    elif model == 'pdp':
                        (_, y_final_test, yp_final_test,
                         _, _, _) = (
                        PDPModel(df_s['x_flow_train'].to_numpy()[0], df_s['x_time_train'].to_numpy()[0], df_s['x_beat_train'].to_numpy()[0], df_s['y_BP_train'].to_numpy()[0],
                        df_s['x_flow_val'].to_numpy()[0], df_s['x_time_val'].to_numpy()[0], df_s['x_beat_val'].to_numpy()[0], df_s['y_BP_val'].to_numpy()[0],
                        df_s['x_flow_test'].to_numpy()[0], df_s['x_time_test'].to_numpy()[0], df_s['x_beat_test'].to_numpy()[0], df_s['y_BP_test'].to_numpy()[0]).model_train(Ew_model=Ew_model, batch=batch, epochs=epochs))

                    y_true.append(y_final_test)
                    y_pred.append(yp_final_test)

                if model == 'conv':
                    name = f'Saved Results\\{model}_{bp_type}_{perc}.pkl'
                else:
                    name = f'Saved Results\\{model}_{Ew_model}E_{bp_type}_{perc}.pkl'
                pd.DataFrame({'True': [y_true], 'Pred': [y_pred]}).to_pickle(name)

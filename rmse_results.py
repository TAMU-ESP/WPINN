import numpy as np
import pandas as pd

class GetMetrics:

    def __init__(self, df):
        self.df = df

    def movmean(self, data, W=10):
        return np.convolve(data, np.ones(W), 'valid') / W

    def extract_sbp(self, BP_waveforms):
        return BP_waveforms.max(axis=1)

    def extract_dbp(self, BP_waveforms):
        return BP_waveforms[:,0]

    def get_error_arr(self, dataset='ring', BP_type='SBP'):
        true, pred = [], []
        if dataset == 'ring':
            s_ind = np.arange(29)[:6]
        else:
            s_ind = np.arange(29)[6:]
        for s in s_ind:
            if BP_type=='SBP':
                sbp_true = GetMetrics.extract_sbp(self, self.df['True'][0][s])
                sbp_pred = GetMetrics.extract_sbp(self, self.df['Pred'][0][s])
                true.extend(list(GetMetrics.movmean(self, sbp_true)))
                pred.extend(list(GetMetrics.movmean(self, sbp_pred)))
            elif BP_type=='DBP':
                dbp_true = GetMetrics.extract_dbp(self, self.df['True'][0][s])
                dbp_pred = GetMetrics.extract_dbp(self, self.df['Pred'][0][s])
                true.extend(list(GetMetrics.movmean(self, dbp_true)))
                pred.extend(list(GetMetrics.movmean(self, dbp_pred)))
            elif BP_type=='WBP':
                true.extend(list(self.df['True'][0][s].flatten()))
                pred.extend(list(self.df['Pred'][0][s].flatten()))
        return np.array(true) - np.array(pred)

for dataset in ['ring', 'hypertensive']:
    for perc in [5.0, 2.5]:
        for bp_type in ['SBP', 'DBP']:
            df_conv = pd.read_pickle(f'Saved Results\\conv_{bp_type}_{perc}.pkl')
            conv_error_arr = GetMetrics(df_conv).get_error_arr(dataset=dataset, BP_type=bp_type)
            for param_type in ['pdp']:
                for E in [2, 3]:
                    df_pinn = pd.read_pickle(f'Saved Results\\{param_type}_{E}E_{bp_type}_{perc}.pkl')
                    pinn_error_arr = GetMetrics(df_pinn).get_error_arr(dataset=dataset, BP_type=bp_type)
                    print(f"{param_type}_{E}E_{bp_type}_{perc}",
                          dataset,
                          round(np.sqrt(np.mean(np.square(conv_error_arr))), 2),
                          round(np.sqrt(np.mean(np.square(pinn_error_arr))), 2))
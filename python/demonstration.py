from auxiliary import get_file
from ruspy.simulation.simulation import simulate
import pickle as pkl
import numpy as np
from zipfile import ZipFile, ZIP_DEFLATED
import os

init_dict = {
    "beta": 0.9999,
    "buses": 1,
    "periods": 85,
    "params": [50, 400],
    "seed": 123,
}
dict_policies = get_file("../pre_processed_data/fixp_results_5000_50_400_4292.pkl")
ev_ml = dict_policies[0.0][0]
ev_95 = dict_policies[0.95][0]
trans_mat = dict_policies[0.0][1]

df_ml = simulate(init_dict, ev_ml, trans_mat)
df_95 = simulate(init_dict, ev_95, trans_mat)

periods_ml = np.array(df_ml["period"], dtype=int)
periods_95 = np.array(df_95["period"], dtype=int)
periods = [periods_ml, periods_95]
states_ml = np.array(df_ml["state"], dtype=int)
states_95 = np.array(df_95["state"], dtype=int)
states = [states_ml, states_95]

for i, df in enumerate([df_ml, df_95]):
    index = np.array(df[df["decision"] == 1].index, dtype=int) + 1
    states[i] = np.insert(states[i], index, 0)
    periods[i] = np.insert(periods[i], index, index - 1)
    print(len(periods[i]), len(states[i]))

file = "demonstration."
pkl.dump((states, periods), open(file + "pkl", "wb"))
ZipFile(file + "zip", "w", ZIP_DEFLATED).write(file + "pkl")
os.remove(file + "pkl")

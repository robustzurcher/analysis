from auxiliary import get_file
from ruspy.simulation.simulation import simulate
import pickle as pkl

init_dict = {
    "beta": 0.9999,
    "buses": 1,
    "periods": 100,
    "params": [50, 400],
    "seed": 123,
}
dict_policies = get_file("../pre_processed_data/fixp_results_5000_50_400_4292.pkl")
ev_ml = dict_policies[0.0][0]
ev_95 = dict_policies[0.95][0]
trans_mat = dict_policies[0.0][1]

df_ml = simulate(init_dict, ev_ml, trans_mat, pool_trans=True)
df_95 = simulate(init_dict, ev_95, trans_mat, pool_trans=True)

pkl.dump((df_ml["state"], df_95["state"]), open("demonstration.pkl", "wb"))

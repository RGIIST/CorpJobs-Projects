# print("Cronbtab running")
# import stable_baselines3
# print("imported")
# with open("cron.txt", "w") as f:
#     f.write("crontab running...")

# import pickle
# from pandas import read_csv

# with open("test_results.pickle", 'rb') as f:
#     res = pickle.load(f)

# # data = read_csv("test_data.csv")
# print(res.keys())
# print(len(res["episode_total_assets"]))
# # print(data)

from stable_baselines3 import A2C
from meta.env_stock_trading.env_stocktrading_np import StockTradingEnv
from pickle import load

with open("test_results.pickle", 'rb') as f:
    prices = load(f)

with open("a2c_results.pickle", 'rb') as f:
    res = load(f)

price_array = prices['price_array']
tech_array = prices['tech_array']
turbulence_array = prices['turbulence_array']

env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": True,
    }

# model = A2C(
#             policy="MlpPolicy",
#             env=StockTradingEnv(config=env_config),
#             tensorboard_log=None,
#             verbose=1,
#             policy_kwargs=None,
#             seed=None,
#             )

# print(dir(model))
print(res.keys())
# print(res['actions'][0])
# print(res['rewards'][0])
# print(res['episode_total_assets'][0])
# print("price_array: ", price_array.shape)
# print("tech_array: ", tech_array.shape)
print("turbulence_array: ", turbulence_array)
print("turbulence_array: ", len(turbulence_array))
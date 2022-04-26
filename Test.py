# source: Deep Reinforcement Learning for Automated Stock Trading: An Ensemble Strategy. ICAIF 2020

import time
from utilities import *
from EnvMultipleStock_train import StockEnvTrain


def run_model() -> None:
    start = time.time()
    filename = 'dow.csv'
    data = preprocess_data(filename)
    print(data.head())
    print(data.size)
    df = data
    initial = True
    train = data_split(df, start=20100101, end=20151231)
    env_train = StockEnvTrain(train)
    obs_trade = env_train.reset()
    R = 0

    for i in range(100):
        action = np.random.randint(0, 3)
        obs_trade, rewards, dones, info = env_train.step(action)
        R += rewards
        print(action, rewards)

    end = time.time()
    print("Strategy took: ", (end - start) / 60, " minutes")
    print("Reword is: ", R, "*1e4")


if __name__ == "__main__":
    run_model()

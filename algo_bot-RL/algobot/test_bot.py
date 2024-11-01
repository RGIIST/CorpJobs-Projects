from env_stocktrading_np import StockTradingEnv
from models import DRLAgent as DRLAgent_sb3
from data_processor import DataProcessor

from config import INDICATORS
from config import TEST_END_DATE
from config import TEST_START_DATE
from config_tickers import DOW_30_TICKER

env = StockTradingEnv


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs,
):
    
    dp = DataProcessor(data_source, **kwargs)
    data = dp.download_data(ticker_list, start_date, end_date, time_interval)
    data = dp.clean_data(data)
    data = dp.add_technical_indicator(data, technical_indicator_list)

    if if_vix:
        data = dp.add_vix(data)
    price_array, tech_array, turbulence_array = dp.df_to_array(data, if_vix)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))
    
    episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
        model_name=model_name, environment=env_instance, cwd=cwd
    )

    return episode_total_assets

if __name__ == "__main__":
    kwargs = (
        {}
    )

    account_value_erl = test(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            ticker_list=DOW_30_TICKER,
            data_source="yahoofinance",
            time_interval="1D",
            technical_indicator_list=INDICATORS,
            drl_lib="stable_baselines3",
            env=env,
            model_name="ppo",
            cwd="./test_ppo.zip",
            # model_name="a2c",
            # cwd="./cnn_a2c.zip",
            net_dimension=512,
            kwargs=kwargs,
        )
    print(account_value_erl)
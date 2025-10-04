import qlib
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.constant import REG_CN

#qlib.init(provider_uri='~/.qlib/qlib_data/my_data', region=REG_CN)
if __name__ == '__main__':
    # qlib.init()
    # market = "csi300"
    qlib.init(provider_uri='~/.qlib/qlib_data/my_data_1', region=REG_CN)
    market = "all"
    benchmark = "SH000300"

    data_handler_config = {
        "start_time": "2019-01-01",
        "end_time": "2025-08-01",
        "fit_start_time": "2019-01-01",
        "fit_end_time": "2023-12-31",
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": ("2019-01-01", "2023-12-31"),
                    "valid": ("2024-01-01", "2025-12-31"),
                    "test": ("2019-01-01", "2023-12-31"),
                },
            },
        },
    }

    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    #start exp
    with R.start(experiment_name="workflow"):
        # train
        R.log_params(**flatten_dict(task))
        model.fit(dataset)

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        # 绘制持仓图表

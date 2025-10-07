import os
from setuptools_scm import get_version
import qlib
from qlib.constant import REG_CN

from qlib.data.dataset.loader import QlibDataLoader

if __name__ == "__main__":
    qlib.init(provider_uri="~/.qlib/qlib_data/my_data_2024", region=REG_CN)
    from qlib.data import D
    qdl = QlibDataLoader(config=(["$close / Ref($close, 10)"], ["RET10"]))
    # ["$open", "$high", "$low", "$close", "$factor"]
    df = qdl.load(instruments="all",start_time="20240101", end_time="20241231")
    print(df.head(50))
    print(D.features(
    ["SH600519"],
    ["(EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9)/$close"],
))
#here = os.getcwd()


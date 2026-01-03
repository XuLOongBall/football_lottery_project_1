import pandas as pd
import config


def load_raw_data():
    """
    读取原始 CSV 数据（自动处理中文编码）
    """
    try:
        df = pd.read_csv(
            config.DATA_FILE,
            encoding="gbk",
            low_memory=False,
            encoding_errors="replace"
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            config.DATA_FILE,
            encoding="latin1",
            low_memory=False
        )

    return df

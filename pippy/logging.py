import logging
import colorlog

def setup_logger():
    # colored log
    log_format = (
        "[%(log_color)s%(levelname)s%(reset)s] "
        "%(log_color)s%(message)s"
        "%(reset)s (%(bold)s%(pathname)s:%(lineno)d%(reset)s)"
    )
    log_colors = {
        'DEBUG':    'bold_cyan',
        'INFO':     'bold_green',
        'WARNING':  'bold_yellow',
        'ERROR':    'bold_red',
        'CRITICAL': 'bold_red',
    }
    formatter = colorlog.ColoredFormatter(log_format, log_colors=log_colors)

    # 创建一个日志器
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # 创建一个流处理器，并为其添加彩色日志格式
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(stream_handler)

    return logger
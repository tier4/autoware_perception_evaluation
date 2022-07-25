import datetime
import logging
import os


def CustomTextFormatter():
    """[summary]
    Custom Formatter
    """
    return logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d %(funcName)s] %(message)s"
    )


class SensitiveWordFilter(logging.Filter):
    """[summary]
    The class to filer sensitive words like password
    """

    def filter(self, record):
        sensitive_words = [
            "password",
            "auth_token",
            "secret",
        ]
        log_message = record.getMessage()
        for word in sensitive_words:
            if word in log_message:
                return False
        return True


def configure_logger(
    log_file_directory: str,
    console_log_level: int = logging.INFO,
    file_log_level: int = logging.INFO,
    logger_name: str = "",
) -> logging.Logger:
    """[summary]
    The function to make logger

    Args:
        log_file_directory (str): The directory path to save log
        console_log_level (int): Log level for console. Defaults to logging.INFO.
        file_log_level (int): Log level for log file. Defaults to logging.INFO.
        logger_name (str): Modname for logger. Defaults to "".
    """
    # make directory
    os.makedirs(log_file_directory, exist_ok=True)

    formatter = CustomTextFormatter()

    logger = logging.getLogger(logger_name)
    logger.handlers.clear()

    logger.addFilter(SensitiveWordFilter())
    logger.setLevel(console_log_level)

    # handler for console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # handler for file
    time = "{0:%Y%m%d_%H%M%S}.log".format(datetime.datetime.now())
    log_file_path = os.path.join(log_file_directory, time)
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setLevel(file_log_level)
    file_formatter = CustomTextFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

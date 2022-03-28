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
    console_log_level=logging.INFO,
    file_log_level=logging.INFO,
) -> logging.Logger:
    """[summary]
    The function to make logger

    Args:
        log_file_directory (str): The directory path to save log
        console_log_level ([type], optional): Log level for console. Defaults to logging.INFO.
        file_log_level ([type], optional): Log level for log file. Defaults to logging.INFO.
        modname ([type], optional): Modname for logger. Defaults to __name__.
    """
    # make directory
    log_directory = os.path.dirname(log_file_directory)
    os.makedirs(log_directory, exist_ok=True)

    formatter = CustomTextFormatter()

    logger = logging.getLogger("")
    logger.addFilter(SensitiveWordFilter())
    logger.setLevel(console_log_level)

    # handler for console
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(console_log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # handler for file
    time = "{0:%Y%m%d_%H%M%S}.txt".format(datetime.datetime.now())
    log_file_path = os.path.join(log_directory, time)
    file_handler = logging.FileHandler(filename=log_file_path, encoding="utf-8")
    file_handler.setLevel(file_log_level)
    file_formatter = CustomTextFormatter()
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger

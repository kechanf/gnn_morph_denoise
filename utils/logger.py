"""
项目统一日志：写入 logs/ 目录，并可选输出到控制台。
使用方式:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("...")
"""
import logging
import os
from datetime import datetime

# 项目根目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")


def get_logger(name: str, level=logging.INFO, console: bool = True) -> logging.Logger:
    """
    获取或创建 logger，日志写入 logs/gnn_project.log，按天追加。
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    log_file = os.path.join(_LOG_DIR, "gnn_project.log")

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=date_fmt)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger


def log_work(entry: str) -> None:
    """
    在 logs/WORKLOG.md 中追加一条工作记录（带日期）。
    """
    os.makedirs(_LOG_DIR, exist_ok=True)
    worklog_path = os.path.join(_LOG_DIR, "WORKLOG.md")
    today = datetime.now().strftime("%Y-%m-%d")
    line = f"\n### {today}\n\n{entry.strip()}\n\n"
    with open(worklog_path, "a", encoding="utf-8") as f:
        f.write(line)

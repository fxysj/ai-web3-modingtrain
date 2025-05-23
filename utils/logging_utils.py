# 日志工具
# utils/logging_utils.py
import os
import logging
import logging.handlers
import sys
import json
import traceback
from datetime import datetime
from typing import Optional, Dict, Any


class CustomFormatter(logging.Formatter):
    """自定义日志格式器，支持彩色输出和结构化日志"""

    # 定义不同日志级别的颜色
    COLORS = {
        'DEBUG': '\033[94m',  # 蓝色
        'INFO': '\033[92m',  # 绿色
        'WARNING': '\033[93m',  # 黄色
        'ERROR': '\033[91m',  # 红色
        'CRITICAL': '\033[95m'  # 紫色
    }
    RESET = '\033[0m'

    # 定义日志格式
    DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    JSON_FORMAT = {
        'timestamp': '%(asctime)s',
        'name': '%(name)s',
        'level': '%(levelname)s',
        'message': '%(message)s'
    }

    def __init__(self, fmt: str = DEFAULT_FORMAT, use_color: bool = True, use_json: bool = False):
        super().__init__()
        self.fmt = fmt
        self.use_color = use_color
        self.use_json = use_json

        # 创建不同级别对应的格式器
        self.formatters = {}
        for level in self.COLORS:
            level_format = self.fmt
            if self.use_color:
                level_format = f"{self.COLORS[level]}{level_format}{self.RESET}"
            self.formatters[level] = logging.Formatter(level_format)

        # JSON格式器
        self.json_formatter = logging.Formatter(json.dumps(self.JSON_FORMAT))

    def format(self, record: logging.LogRecord) -> str:
        if self.use_json:
            return self.json_formatter.format(record)

        # 根据日志级别选择格式器
        formatter = self.formatters.get(record.levelname, self.formatters['INFO'])
        return formatter.format(record)


class ErrorTracker:
    """错误跟踪器，用于捕获和记录异常信息"""

    @staticmethod
    def log_exception(logger: logging.Logger, message: str = "An error occurred",
                      exc_info: Optional[Any] = None) -> None:
        """记录详细的异常信息"""
        if not exc_info:
            exc_info = sys.exc_info()

        if not exc_info or exc_info[0] is None:
            logger.error(message)
            return

        # 获取异常类型、值和堆栈跟踪
        exc_type, exc_value, exc_traceback = exc_info

        # 构建异常信息
        error_info = {
            'error_type': exc_type.__name__,
            'error_message': str(exc_value),
            'timestamp': datetime.now().isoformat(),
            'stack_trace': traceback.format_exc()
        }

        # 记录错误信息
        logger.error(f"{message}: {json.dumps(error_info, indent=2)}")

    @staticmethod
    def get_stack_trace() -> str:
        """获取当前堆栈跟踪信息"""
        return traceback.format_exc()


class LogMetricsFilter(logging.Filter):
    """日志过滤器，用于捕获和记录性能指标"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 检查是否包含指标数据
        if hasattr(record, 'metrics'):
            # 可以在这里将指标数据发送到监控系统
            print(f"Metrics captured: {record.metrics}")
        return True


class LoggerBuilder:
    """日志构建器，用于创建配置好的logger实例"""

    @staticmethod
    def setup_logger(name: str, log_file: Optional[str] = None,
                     level: int = logging.INFO, use_color: bool = True,
                     use_json: bool = False, max_size: int = 10 * 1024 * 1024,
                     backup_count: int = 5) -> logging.Logger:
        """
        设置并返回一个配置好的logger

        Args:
            name: 日志名称
            log_file: 日志文件路径，如果为None则只输出到控制台
            level: 日志级别，默认为INFO
            use_color: 是否使用彩色输出
            use_json: 是否使用JSON格式
            max_size: 日志文件最大大小（字节），用于日志轮转
            backup_count: 保留的旧日志文件数量
        """
        # 创建logger
        logger = logging.getLogger(name)

        # 避免重复添加处理器
        if not logger.handlers:
            # 设置日志级别
            logger.setLevel(level)

            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # 设置格式器
            formatter = CustomFormatter(use_color=use_color, use_json=use_json)
            console_handler.setFormatter(formatter)

            # 添加控制台处理器
            logger.addHandler(console_handler)

            # 如果指定了日志文件，添加文件处理器
            if log_file:
                # 创建日志目录
                log_dir = os.path.dirname(log_file)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)

                # 使用RotatingFileHandler实现日志轮转
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_size, backupCount=backup_count
                )
                file_handler.setLevel(level)

                # 文件日志不使用颜色
                file_formatter = CustomFormatter(use_color=False, use_json=use_json)
                file_handler.setFormatter(file_formatter)

                # 添加文件处理器
                logger.addHandler(file_handler)

            # 添加指标过滤器
            logger.addFilter(LogMetricsFilter())

            # 设置异常钩子，确保未捕获的异常也被记录
            sys.excepthook = lambda exc_type, exc_value, exc_traceback: \
                ErrorTracker.log_exception(logger, "Uncaught exception", (exc_type, exc_value, exc_traceback))

        return logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """获取已配置的logger"""
        return logging.getLogger(name)


# 快速创建默认logger的函数
def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """快速设置并返回一个配置好的logger"""
    return LoggerBuilder.setup_logger(name, log_file, level)


# 示例使用
if __name__ == "__main__":
    # 创建一个控制台输出的logger
    console_logger = setup_logger("console_logger")
    console_logger.info("This is an info message")
    console_logger.warning("This is a warning message")
    console_logger.error("This is an error message")

    # 创建一个输出到文件的logger
    file_logger = setup_logger("file_logger", "example.log")
    file_logger.info("This message will be logged to file")

    # 演示异常捕获
    try:
        result = 1 / 0
    except Exception as e:
        ErrorTracker.log_exception(file_logger, "Division by zero error")

    # 演示结构化日志
    structured_logger = setup_logger("structured_logger", use_json=True)
    structured_logger.info({
        "action": "user_login",
        "user_id": "12345",
        "timestamp": datetime.now().isoformat()
    })
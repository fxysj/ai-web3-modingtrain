# 监控部署
# scripts/monitor_deployment.py
import argparse
import requests
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from prometheus_client import start_http_server, Summary, Counter, Gauge
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import smtplib

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("model_monitor")

# 定义Prometheus指标
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
REQUEST_COUNT = Counter('request_total', 'Total count of requests', ['status_code', 'endpoint'])
LATENCY_GAUGE = Gauge('request_latency_seconds', 'Request latency in seconds', ['endpoint'])
ERROR_COUNT = Counter('request_errors_total', 'Total count of errors', ['endpoint', 'error_type'])
PREDICTION_QUALITY = Gauge('prediction_quality', 'Quality metric of predictions', ['metric'])


class ModelMonitor:
    def __init__(self, config_path):
        """初始化模型监控器"""
        self.config = self._load_config(config_path)
        self.api_endpoint = self.config["api_endpoint"]
        self.auth_token = self.config.get("auth_token")
        self.monitoring_interval = self.config.get("monitoring_interval", 60)
        self.alert_thresholds = self.config.get("alert_thresholds", {})
        self.history = []

        # 初始化邮件配置
        self.email_config = self.config.get("email_config", {})
        self.enable_email_alerts = self.email_config.get("enable", False)

        # 初始化Prometheus指标服务器
        self._init_prometheus_metrics()

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        port = self.config.get("prometheus_port", 8000)
        start_http_server(port)
        logger.info(f"Prometheus metrics server started on port {port}")

    def _send_email_alert(self, subject, message, images=None):
        """发送邮件警报"""
        if not self.enable_email_alerts:
            return

        try:
            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = self.email_config["sender"]
            msg['To'] = ", ".join(self.email_config["recipients"])
            msg['Subject'] = subject

            # 添加消息正文
            msg.attach(MIMEText(message, 'plain'))

            # 添加图像附件
            if images:
                for image_name, image_data in images.items():
                    image = MIMEImage(image_data)
                    image.add_header('Content-Disposition', 'attachment', filename=image_name)
                    msg.attach(image)

            # 发送邮件
            with smtplib.SMTP(self.email_config["smtp_server"], self.email_config["smtp_port"]) as server:
                server.starttls()
                server.login(self.email_config["sender"], self.email_config["password"])
                server.sendmail(
                    self.email_config["sender"],
                    self.email_config["recipients"],
                    msg.as_string()
                )

            logger.info(f"Email alert sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _check_alert_conditions(self, metrics):
        """检查警报条件"""
        alerts = []

        # 检查响应时间
        if "avg_response_time" in metrics:
            threshold = self.alert_thresholds.get("response_time", 2.0)
            if metrics["avg_response_time"] > threshold:
                alerts.append(f"高响应时间: {metrics['avg_response_time']:.2f}秒 (阈值: {threshold}秒)")

        # 检查错误率
        if "error_rate" in metrics:
            threshold = self.alert_thresholds.get("error_rate", 0.05)
            if metrics["error_rate"] > threshold:
                alerts.append(f"高错误率: {metrics['error_rate']:.2%} (阈值: {threshold:.2%})")

        # 检查预测质量
        if "prediction_quality" in metrics:
            threshold = self.alert_thresholds.get("prediction_quality", 0.8)
            if metrics["prediction_quality"] < threshold:
                alerts.append(f"预测质量下降: {metrics['prediction_quality']:.2f} (阈值: {threshold})")

        return alerts

    def _collect_metrics(self):
        """收集模型API的性能指标"""
        headers = {"Authorization": f"Bearer {self.auth_token}"} if self.auth_token else {}

        # 准备测试请求
        test_prompts = [
            "分析以太坊链上最近一周的交易趋势，特别是DeFi协议的表现。",
            "预测比特币价格在未来一个月的走势，并分析主要影响因素。",
            "解释Uniswap V3与V2相比的主要创新点和优势。",
        ]

        results = []
        errors = 0

        # 测试API
        for prompt in test_prompts:
            payload = {"prompt": prompt, "temperature": 0.7, "max_tokens": 512}

            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.api_endpoint}/predict",
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                end_time = time.time()

                # 记录请求时间
                request_time = end_time - start_time
                LATENCY_GAUGE.labels(endpoint="/predict").set(request_time)
                REQUEST_TIME.observe(request_time)

                # 检查响应状态
                if response.status_code == 200:
                    results.append({
                        "prompt": prompt,
                        "response": response.json(),
                        "response_time": request_time,
                        "success": True
                    })
                    REQUEST_COUNT.labels(status_code="200", endpoint="/predict").inc()
                else:
                    errors += 1
                    results.append({
                        "prompt": prompt,
                        "error": response.text,
                        "response_time": request_time,
                        "success": False
                    })
                    REQUEST_COUNT.labels(status_code=str(response.status_code), endpoint="/predict").inc()
                    ERROR_COUNT.labels(endpoint="/predict", error_type="API_ERROR").inc()

                    logger.error(f"API请求失败: {response.status_code}, {response.text}")

            except Exception as e:
                errors += 1
                results.append({
                    "prompt": prompt,
                    "error": str(e),
                    "success": False
                })
                ERROR_COUNT.labels(endpoint="/predict", error_type="EXCEPTION").inc()

                logger.error(f"API请求异常: {e}")

        # 计算指标
        avg_response_time = np.mean([r["response_time"] for r in results if r["success"]])
        error_rate = errors / len(test_prompts)

        # 模拟预测质量评估（实际应用中应使用真实评估方法）
        prediction_quality = np.random.uniform(0.7, 0.95)  # 模拟质量分数
        PREDICTION_QUALITY.labels(metric="f1_score").set(prediction_quality)

        metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_response_time": avg_response_time,
            "error_rate": error_rate,
            "request_count": len(test_prompts),
            "success_count": len(test_prompts) - errors,
            "prediction_quality": prediction_quality,
            "results": results
        }

        self.history.append(metrics)

        # 保留最近100个历史记录
        if len(self.history) > 100:
            self.history.pop(0)

        return metrics

    def _generate_performance_report(self, metrics):
        """生成性能报告"""
        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "api_endpoint": self.api_endpoint,
            "performance_metrics": metrics,
            "alert_status": self._check_alert_conditions(metrics)
        }

        return report

    def _plot_metrics_history(self):
        """绘制指标历史图表"""
        if not self.history:
            return None

        # 提取数据
        timestamps = [h["timestamp"] for h in self.history]
        response_times = [h["avg_response_time"] for h in self.history]
        error_rates = [h["error_rate"] for h in self.history]
        quality_scores = [h.get("prediction_quality", 0) for h in self.history]

        # 创建图表
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # 响应时间图表
        ax1.plot(timestamps, response_times, 'b-')
        ax1.set_title('平均响应时间')
        ax1.set_xlabel('时间')
        ax1.set_ylabel('秒')
        ax1.tick_params(axis='x', rotation=45)

        # 错误率图表
        ax2.plot(timestamps, error_rates, 'r-')
        ax2.set_title('错误率')
        ax2.set_xlabel('时间')
        ax2.set_ylabel('比例')
        ax2.tick_params(axis='x', rotation=45)

        # 预测质量图表
        ax3.plot(timestamps, quality_scores, 'g-')
        ax3.set_title('预测质量')
        ax3.set_xlabel('时间')
        ax3.set_ylabel('分数')
        ax3.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 保存图表
        import io
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)

        return img_buffer.getvalue()

    def run(self):
        """运行监控循环"""
        logger.info(f"Starting model monitoring for {self.api_endpoint}")

        try:
            while True:
                logger.info("Collecting metrics...")
                metrics = self._collect_metrics()
                report = self._generate_performance_report(metrics)

                # 记录报告
                logger.info(f"Performance report: {json.dumps(report, indent=2)}")

                # 检查警报
                alerts = report["alert_status"]
                if alerts:
                    alert_message = "\n".join([f"ALERT: {alert}" for alert in alerts])
                    logger.warning(alert_message)

                    # 生成性能图表
                    chart_image = self._plot_metrics_history()

                    # 发送邮件警报
                    self._send_email_alert(
                        f"[模型监控] 发现 {len(alerts)} 个异常",
                        f"时间: {report['timestamp']}\n\n"
                        f"API 端点: {self.api_endpoint}\n\n"
                        f"异常详情:\n{alert_message}\n\n"
                        f"性能指标:\n{json.dumps(metrics, indent=2, ensure_ascii=False)}",
                        {"performance_chart.png": chart_image} if chart_image else None
                    )

                # 等待下一个监控周期
                logger.info(f"Waiting for {self.monitoring_interval} seconds...")
                time.sleep(self.monitoring_interval)

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Monitor a deployed Web3.0 model API")
    parser.add_argument("--config", type=str, default="config/monitoring_config.json",
                        help="配置文件路径")
    args = parser.parse_args()

    # 启动监控
    monitor = ModelMonitor(args.config)
    monitor.run()


if __name__ == "__main__":
    main()
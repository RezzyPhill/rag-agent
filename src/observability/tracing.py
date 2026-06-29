import structlog


# Configure structlog once at application startup.
def configure_logging() -> None:
    structlog.configure(
        processors=[
            # {"level": "info"} to every log record
            structlog.stdlib.add_log_level,
            # {"timestamp": "2026-06-29T04:17:00Z"} in ISO 8601
            structlog.processors.TimeStamper(fmt="iso"),
            # renders the whole record as a single JSON string to stdout
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


# Call once per request in the API layer then pass the logger down to the orchestrator
def get_logger(trace_id: str | None = None) -> structlog.BoundLogger:
    logger = structlog.get_logger()
    if trace_id:
        logger = logger.bind(trace_id=trace_id)
    return logger

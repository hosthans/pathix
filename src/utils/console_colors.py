import logging

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[92m",
        logging.ERROR: "\033[91m",
        logging.DEBUG: "\033[94m",
        logging.WARNING: "\033[33m",
    }
    RESET = "\033[0m"

    def format(self, record):
        """Format a log record with an ANSI color based on its level."""
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"
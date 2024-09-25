import logging

LOGGER = logging.getLogger(__name__)


def test_eggs():
    LOGGER.info("info")
    LOGGER.warning("warning")
    LOGGER.error("error")
    LOGGER.critical("critical")
    assert True

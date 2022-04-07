G_LOGGER = None

def set_logger(logger):
    global G_LOGGER
    G_LOGGER = logger

def log_print(text):
    global G_LOGGER
    print(text)
    if G_LOGGER is not None:
        G_LOGGER.log(text)

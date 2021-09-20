import logging
import tqdm
import os
from time import strftime
from datetime import datetime
import sys
from train_doc import train_nn_doc
from train_ss import train_nn_sent
from train_nli import train_nn_nli
import config


def main(models_list):
    for model_name in models_list:
        if 'doc' in model_name:
            call_trainer(train_nn_doc, model_name)

        elif 'ss' in model_name:
            call_trainer(train_nn_sent, model_name)

        else:
            call_trainer(train_nn_nli, model_name)


if __name__ == '__main__':
    models = ['nn_doc'] # can have nn_doc, nn_ss, nn_nli
    n_models = 3

    log_dir = config.PRO_ROOT / "logs"
    date_dir = strftime('%d-%m-%Y')
    time_dir = strftime('%X')
    log_path = os.path.join(log_dir, date_dir, time_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"train.log")

    file_formatter = logging.Formatter(
        fmt='%(levelname)s::%(asctime)s::%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    stream_formatter = logging.Formatter(
        fmt='%(message)s'
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    call_trainer = lambda x, y : x(y, logger, date_dir, time_dir)

    main(models)


    #train_nn_doc(9000, 1, 'nn_doc')
    #train_nn_sent(57167, 6, 'nn_ss')
    #train_nn_sent(77083, 7, 'nn_ss1')
    #train_nn_sent(58915, 7, 'nn_ss2')
    #train_nn_nli(77000, 11, 'nn_nli')

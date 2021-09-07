from nn_doc_retrieval.nn_doc_model import *
#from sentence_retrieval.simple_nnmodel import *
# from nli.mesim_wn_simi_v1_2 import *

import logging
import tqdm
import os
from time import strftime
from datetime import datetime
import sys

def train_nn_doc(model_name):
    logger.info(f"training doc model with name {model_name}")

    num_epoch = 10
    seed = 12
    batch_size = 64
    dev_batch_size = 128
    lazy = True
    torch.manual_seed(seed)
    contain_first_sentence = True
    pn_ratio = 1.0
    # keep_neg_sample_prob = 0.4
    # sample_prob_decay = 0.05

    logger.info(f"""training parameters:
epoch number = {num_epoch}
seed = {seed}
batch_size = {batch_size}
dev_batch_size = {dev_batch_size}
lazy = {lazy}
contain_first_sentence = {contain_first_sentence}
pn_ration = {pn_ratio}""")

    dev_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_08_22_13:03:04_r/doc_retr_1_shared_task_dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_09_02_16:31:02_r/doc_retr_1_train.jsonl"

    logger.info(f"""dev file path = {dev_upstream_file}
train file path = {train_upstream_file}""")

    dev_data_list = common.load_jsonl(dev_upstream_file)
    train_data_list = common.load_jsonl(train_upstream_file)

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    cursor = fever_db.get_cursor()
    complete_upstream_dev_data = disamb.sample_disamb_inference(dev_data_list, cursor,
                                                                contain_first_sentence=contain_first_sentence)
    logger.info(f"Dev size: {len(complete_upstream_dev_data)}")
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=dev_batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value

    vocab.get_index_to_token_vocabulary('selection_labels')

    logger.info(vocab.get_token_to_index_vocabulary('selection_labels'))
    logger.info(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    model.display()
    model.to(device)

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{model_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    logger.info("start training")

    for i_epoch in range(num_epoch):
        start = datetime.now()
        logger.info(f"begin epoch {i_epoch}")
        logger.info("Resampling...")
        # Resampling
        complete_upstream_train_data = disamb.sample_disamb_training_v0(train_data_list,
                                                                        cursor, pn_ratio, contain_first_sentence,
                                                                        only_found=False)
        random.shuffle(complete_upstream_train_data)
        logger.info(f"Sample Prob.:  {pn_ratio}")

        logger.info(f"Sampled_length: {len(complete_upstream_train_data)}")
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm.tqdm(enumerate(train_iter), desc=f"epoch: {i_epoch + 1} / {num_epoch}, iteration: "):
            model.train()
            out = model(batch)
            y = batch['selection_label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch > 5:
                mod = 1000
            else:
                mod = 500

            if iteration % mod == 0:
                logger.info("Evaluating ...")

                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                disamb.enforce_disabuigation_into_retrieval_result_v0(complete_upstream_dev_data,
                                                                      dev_data_list)
                oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

                logger.info(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
                logger.info(f"Strict score: {oracle_score}")
                logger.info(f"Eval Tracking score: {oracle_score}")

                need_save = False
                if oracle_score > best_dev:
                    best_dev = oracle_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{oracle_score}|pr:{pr}|rec:{rec}|f1:{f1})'
                    )

                    torch.save(model.state_dict(), save_path)

        #
        end = datetime.now()
        logger.info(f"finish epoch {i_epoch} in {end - start} time")
        logger.info("Epoch Evaluation...")
        eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
        complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

        disamb.enforce_disabuigation_into_retrieval_result_v0(complete_upstream_dev_data,
                                                              dev_data_list)
        oracle_score, pr, rec, f1 = c_scorer.fever_doc_only(dev_data_list, dev_data_list, max_evidence=5)

        logger.info(f"Dev(raw_acc/pr/rec/f1):{oracle_score}/{pr}/{rec}/{f1}")
        logger.info(f"Strict score: {oracle_score}")
        logger.info(f"Eval Tracking score: {oracle_score}")

        need_save = False
        if oracle_score > best_dev:
            best_dev = oracle_score
            need_save = True

        if need_save:
            save_path = os.path.join(
                file_path_prefix,
                f'i({iteration})_epoch({i_epoch})_e'
                f'(tra_score:{oracle_score}|pr:{pr}|rec:{rec}|f1:{f1})'
            )

            torch.save(model.state_dict(), save_path)


def train_nn_sent(model_name):
    num_epoch = 10
    seed = 12
    batch_size = 64
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 0.4
    sample_prob_decay = 0.05

    dev_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_08_22_13:03:04_r/doc_retr_1_shared_task_dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_09_02_16:31:02_r/doc_retr_1_train.jsonl"

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_upstream_file, pred=True)
    logger.info("Dev size:", len(complete_upstream_dev_data))
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)
    dev_biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    # THis is important
    vocab.add_token_to_namespace("true", namespace="selection_labels")
    vocab.add_token_to_namespace("false", namespace="selection_labels")
    vocab.add_token_to_namespace("hidden", namespace="selection_labels")
    vocab.change_token_with_index_to_namespace("hidden", -2, namespace='selection_labels')
    # Label value

    vocab.get_index_to_token_vocabulary('selection_labels')

    logger.info(vocab.get_token_to_index_vocabulary('selection_labels'))
    logger.info(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)
    dev_biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  embedding_dim=300, max_l=160, num_of_class=2)

    model.display()
    model.to(device)

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{model_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()

    for i_epoch in trange(num_epoch, desc="epoch"):
        logger.info("Resampling...")
        # Resampling
        complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_upstream_file, pred=False)
        logger.info("Sample Prob.:", keep_neg_sample_prob)
        filtered_train_data = post_filter(complete_upstream_train_data, keep_prob=keep_neg_sample_prob,
                                          seed=12 + i_epoch)
        # Change the seed to avoid duplicate sample...
        keep_neg_sample_prob -= sample_prob_decay
        if keep_neg_sample_prob <= 0:
            keep_neg_sample_prob = 0.005
        logger.info("Sampled_length:", len(filtered_train_data))
        sampled_train_instances = train_fever_data_reader.read(filtered_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in trange(enumerate(train_iter), desc="iteration"):
            model.train()
            out = model(batch)
            y = batch['selection_label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 7:
                mod = 20000
            else:
                mod = 8000

            if iteration % mod == 0:
                eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

                dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
                eval_mode = {'check_sent_id_correct': True, 'standard': True}
                strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, dev_results_list,
                                                                            mode=eval_mode, verbose=False)
                total = len(dev_results_list)
                hit = eval_mode['check_sent_id_correct_hits']
                tracking_score = hit / total

                logger.info(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
                logger.info("Strict score:", strict_score)
                logger.info(f"Eval Tracking score:", f"{tracking_score}")

                need_save = False
                if tracking_score > best_dev:
                    best_dev = tracking_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_'
                        f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})'
                    )

                    torch.save(model.state_dict(), save_path)


        logger.info("Epoch Evaluation...")
        eval_iter = dev_biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
        complete_upstream_dev_data = hidden_eval(model, eval_iter, complete_upstream_dev_data)

        dev_results_list = score_converter_v0(config.T_FEVER_DEV_JSONL, complete_upstream_dev_data)
        eval_mode = {'check_sent_id_correct': True, 'standard': True}
        strict_score, acc_score, pr, rec, f1 = c_scorer.fever_score(dev_results_list, dev_results_list,
                                                                    mode=eval_mode, verbose=False)
        total = len(dev_results_list)
        hit = eval_mode['check_sent_id_correct_hits']
        tracking_score = hit / total

        logger.info(f"Dev(raw_acc/pr/rec/f1):{acc_score}/{pr}/{rec}/{f1}/")
        logger.info("Strict score:", strict_score)
        logger.info(f"Eval Tracking score:", f"{tracking_score}")

        if tracking_score > best_dev:
            best_dev = tracking_score

            save_path = os.path.join(
                file_path_prefix,
                f'i({iteration})_epoch({i_epoch})_'
                f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})_epoch'
            )

            torch.save(model.state_dict(), save_path)


def train_nn_nli(model_name):
    num_epoch = 12
    seed = 12
    batch_size = 32
    lazy = True
    dev_prob_threshold = 0.5
    train_prob_threshold = 0.35
    train_sample_top_k = 12

    logger.info("Dev prob threshold:", dev_prob_threshold)
    logger.info("Train prob threshold:", train_prob_threshold)
    logger.info("Train sample top k:", train_sample_top_k)

    dev_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                               "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    train_upstream_sent_list = common.load_jsonl(config.RESULT_PATH /
                                                 "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    p_dict = wn_persistent_api.persistence_load()

    logger.info("Building Prob Dicts...")
    train_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/train_sent.jsonl")
    remaining_sent_list = common.load_jsonl(
        config.RESULT_PATH / "sent_retri_nn/remaining_training_cache/remain_train_sent.jsonl")
    dev_sent_list = common.load_jsonl(config.RESULT_PATH / "sent_retri_nn/2018_07_20_15:17:59_r/dev_sent.jsonl")

    selection_dict = paired_selection_score_dict(train_sent_list)
    selection_dict = paired_selection_score_dict(dev_sent_list, selection_dict)
    selection_dict = paired_selection_score_dict(remaining_sent_list, selection_dict)

    upstream_dev_list = threshold_sampler(config.T_FEVER_DEV_JSONL, dev_upstream_sent_list,
                                          prob_threshold=dev_prob_threshold, top_n=5)

    dev_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)
    train_fever_data_reader = WNSIMIReader(token_indexers=token_indexers, lazy=lazy, wn_p_dict=p_dict, max_l=360)

    complete_upstream_dev_data = select_sent_with_prob_for_eval(config.T_FEVER_DEV_JSONL, upstream_dev_list,
                                                                selection_dict, tokenized=True)
    dev_instances = dev_fever_data_reader.read(complete_upstream_dev_data)

    # Load Vocabulary
    biterator = BasicIterator(batch_size=batch_size)

    vocab, weight_dict = load_vocab_embeddings(config.DATA_ROOT / "vocab_cache" / "nli_basic")
    vocab.change_token_with_index_to_namespace('hidden', -2, namespace='labels')

    logger.info(vocab.get_token_to_index_vocabulary('labels'))
    logger.info(vocab.get_vocab_size('tokens'))

    biterator.index_with(vocab)

    # Build Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)
    device_num = -1 if device.type == 'cpu' else 0

    model = Model(rnn_size_in=(1024 + 300 + dev_fever_data_reader.wn_feature_size,
                               1024 + 450 + dev_fever_data_reader.wn_feature_size),
                  rnn_size_out=(450, 450),
                  weight=weight_dict['glove.840B.300d'],
                  vocab_size=vocab.get_vocab_size('tokens'),
                  mlp_d=900,
                  embedding_dim=300, max_l=300)

    logger.info("Model Max length:", model.max_l)
    model.display()
    model.to(device)

    # Create Log File
    file_path_prefix, date = save_tool.gen_file_prefix(f"{model_name}")
    # Save the source code.
    script_name = os.path.basename(__file__)
    with open(os.path.join(file_path_prefix, script_name), 'w') as out_f, open(__file__, 'r') as it:
        out_f.write(it.read())
        out_f.flush()
    # Save source code end.

    best_dev = -1
    iteration = 0

    start_lr = 0.0002
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
    criterion = nn.CrossEntropyLoss()


    for i_epoch in range(num_epoch):

        logger.info("Resampling...")
        # Resampling
        train_data_with_candidate_sample_list = \
            threshold_sampler(config.T_FEVER_TRAIN_JSONL, train_upstream_sent_list, train_prob_threshold,
                              top_n=train_sample_top_k)

        complete_upstream_train_data = adv_simi_sample_with_prob_v1_0(config.T_FEVER_TRAIN_JSONL,
                                                                      train_data_with_candidate_sample_list,
                                                                      selection_dict,
                                                                      tokenized=True)

        logger.info("Sample data length:", len(complete_upstream_train_data))
        sampled_train_instances = train_fever_data_reader.read(complete_upstream_train_data)

        train_iter = biterator(sampled_train_instances, shuffle=True, num_epochs=1, cuda_device=device_num)
        for i, batch in tqdm(enumerate(train_iter)):
            model.train()
            out = model(batch)
            y = batch['label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch <= 6:
                # mod = 5000
                mod = 10000
            else:
                mod = 500

            if iteration % mod == 0:
                eval_iter = biterator(dev_instances, shuffle=False, num_epochs=1, cuda_device=device_num)
                dev_score, dev_loss = full_eval_model(model, eval_iter, criterion, complete_upstream_dev_data)

                logger.info(f"Dev:{dev_score}/{dev_loss}")

                need_save = False
                if dev_score > best_dev:
                    best_dev = dev_score
                    need_save = True

                if need_save:
                    save_path = os.path.join(
                        file_path_prefix,
                        f'i({iteration})_epoch({i_epoch})_dev({dev_score})_loss({dev_loss})_seed({seed})'
                    )

                    torch.save(model.state_dict(), save_path)


        # Save some cache wordnet feature.
        wn_persistent_api.persistence_update(p_dict)


def main(models_list):
    for model_name in models_list:
        if 'doc' in model_name:
            train_nn_doc(model_name)

        elif 'ss' in model_name:
            train_nn_sent(model_name)

        else:
            train_nn_nli(model_name)


if __name__ == '__main__':
    models = ['nn_doc'] # can have nn_doc, nn_ss, nn_nli

    log_dir = config.PRO_ROOT / "logs"
    date_dir = strftime('%d-%m-%Y')
    time_dir = strftime('%X')
    log_path = os.path.join(log_dir, date_dir, time_dir)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"train.log")

    formatter = logging.Formatter(
        fmt='%(levelname)s::%(asctime)s::%(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
    )

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    stream_handler.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    main(models)

    #train_nn_doc(9000, 1, 'nn_doc')
    #train_nn_sent(57167, 6, 'nn_ss')
    #train_nn_sent(77083, 7, 'nn_ss1')
    #train_nn_sent(58915, 7, 'nn_ss2')
    #train_nn_nli(77000, 11, 'nn_nli')

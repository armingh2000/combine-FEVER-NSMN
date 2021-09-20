from sentence_retrieval.simple_nnmodel import *


def train_nn_sent(model_name, logger, date_dir, time_dir):
    logger.info(f"training sentence model with name {model_name}")

    num_epoch = 10
    seed = 12
    batch_size = 64
    lazy = True
    torch.manual_seed(seed)
    keep_neg_sample_prob = 0.4
    sample_prob_decay = 0.05

    logger.info(f"""training parameters:
epoch number = {num_epoch}
seed = {seed}
batch_size = {batch_size}
lazy = {lazy}
keep_neg_sample_prob = {keep_neg_sample_prob}
sample_prob_decay = {sample_prob_decay}""")

    dev_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_08_22_13:03:04_r/doc_retr_1_shared_task_dev.jsonl"
    train_upstream_file = config.RESULT_PATH / "pipeline_r_aaai_doc/2021_09_02_16:31:02_r/doc_retr_1_train.jsonl"

    logger.info(f"""dev file path = {dev_upstream_file}
train file path = {train_upstream_file}""")

    dev_data_list = common.load_jsonl(dev_upstream_file)[:1000]
    train_data_list = common.load_jsonl(train_upstream_file)[:2000]

    # Prepare Data
    token_indexers = {
        'tokens': SingleIdTokenIndexer(namespace='tokens'),  # This is the raw tokens
        'elmo_chars': ELMoTokenCharactersIndexer(namespace='elmo_characters')  # This is the elmo_characters
    }

    train_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)
    dev_fever_data_reader = SSelectorReader(token_indexers=token_indexers, lazy=lazy, max_l=180)

    complete_upstream_dev_data = get_full_list(config.T_FEVER_DEV_JSONL, dev_data_list, pred=True)
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
    file_path_prefix = os.path.join(file_path_prefix, date_dir, time_dir)
    if not os.path.exists(file_path_prefix):
        os.makedirs(file_path_prefix)
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
    saved_models = []

    for i_epoch in trange(num_epoch, desc="epoch"):
        start = datetime.now()
        logger.info(f"begin epoch {i_epoch}")
        logger.info("Resampling...")
        # Resampling
        complete_upstream_train_data = get_full_list(config.T_FEVER_TRAIN_JSONL, train_data_list, pred=False)
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
        for i, batch in tqdm.tqdm(enumerate(train_iter), desc=f"epoch: {i_epoch} / {num_epoch - 1}, iteration: "):
            model.train()
            out = model(batch)
            y = batch['selection_label']

            loss = criterion(out, y)

            # No decay
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            if i_epoch > 7:
                mod = 20000
            else:
                mod = 8000

            if iteration % mod == 0:
                logger.info("Evaluating ...")

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
                    if len(saved_models) == n_models:
                        os.remove(os.path.join(file_path_prefix, saved_models[0]))
                        logger.info(f"remove model {saved_models.pop(0)} to keep {n_models} limits")

                    model_file = f'i({iteration})_epoch({i_epoch})_' \
                        f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})'

                    save_path = os.path.join(
                        file_path_prefix,
                        model_file
                    )

                    torch.save(model.state_dict(), save_path)
                    saved_models.append(model_file)

                    logger.info(f"saved model in the middle of epoch {i_epoch}: {model_file}")

        end = datetime.now()
        logger.info(f"finish epoch {i_epoch} in {end - start} time")
        logger.info("Epoch Evaluation...")
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

        if tracking_score > best_dev or i_epoch == num_epoch - 1:
            best_dev = tracking_score

            if len(saved_models) == n_models:
                os.remove(os.path.join(file_path_prefix, saved_models[0]))
                logger.info(f"remove model {saved_models.pop(0)} to keep {n_models} limits")

            model_file = f'i({iteration})_epoch({i_epoch})_' \
                f'(tra_score:{tracking_score}|raw_acc:{acc_score}|pr:{pr}|rec:{rec}|f1:{f1})'

            save_path = os.path.join(
                file_path_prefix,
                model_file
            )

            torch.save(model.state_dict(), save_path)
            saved_models.append(model_file)

            logger.info(f"saved model at the end of epoch {i_epoch}: {model_file}")

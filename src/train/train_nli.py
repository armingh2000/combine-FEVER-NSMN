from nli.mesim_wn_simi_v1_2 import *
from datetime import datetime


def train_nn_nli(model_name, logger, date_dir, time_dir):
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

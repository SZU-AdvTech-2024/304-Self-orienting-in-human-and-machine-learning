param_abbreviations = {
    'seed': '', 'verbose': '',
    # DQN
    'gamma': 'g', 'learning_rate': 'lr', 'buffer_size': 's_buf', 'learning_starts': 'lrst',
    'exploration_fraction': 'exf', 'exploration_final_eps': 'exp_f_eps',
    'exploration_initial_eps': 'ieps', 'train_freq': 'trfrq', 'batch_size': 'b_s',
    'double_q': 'dq', 'prioritized_replay_alpha': 'pra', 'prioritized_replay_beta0': 'prb',
    'n_timesteps': 'n_ts', 'prioritized_replay_beta_iters': 'prbi',
    'prioritized_replay_eps': 'preps', 'prioritized_replay': 'pr',
    'param_noise': 'prn', 'policy': 'plc', 'tensorboard_log': '', '_init_setup_model': '', 'policy_kwargs': '',
    'full_tensorboard_log': '', 'n_cpu_tf_sess': '', 'kwargs': '', 'target_network_update_freq': 'netuf',
    # PPO2
    'n_steps': 'n_s', 'ent_coef': 'ent', 'vf_coef': 'vf', 'max_grad_norm': 'mxgn',
    'lam': 'lam', 'nminibatches': 'nmb', 'noptepochs': 'nepoc', 'cliprange': 'cr', 'cliprange_vf': 'cr_vf',
    # TRPO
    'timesteps_per_batch': 'tspb', 'max_kl': 'maxkl', 'cg_iters': 'cgite',
    'entcoeff': 'ent', 'cg_damping': 'cgd', 'vf_stepsize': 'vfs', 'vf_iters': 'vfite',
    # GAIL
    'expert_dataset': '', 'hidden_size_adversary': 'hsa', 'adversary_entcoeff': 'aec',
    'g_step': 'g_step', 'd_step': 'd_step', 'd_stepsize': 'd_ss',
    # HER
    'model_class': 'model_class', 'n_sampled_goal': 'nsg',
    'goal_selection_strategy': 'gss', 'args': '',
    # ACKTR
    'nprocs': 'nprocs', 'vf_fisher_coef': 'vffc', 'kfac_clip': 'kfacc', 'lr_schedule': 'lrsc',
    'async_eigen_decomp': 'eigen_dec', 'kfac_update': 'kfac_upd', 'gae_lambda': 'gae_lmd',
    # A2C
    'alpha': 'alpha', 'momentum': 'mmt', 'epsilon': 'epsilon',
    # ACER
    'q_coef': 'q', 'rprop_alpha': 'rprop_a', 'rprop_epsilon': 'rprop_e', 'replay_ratio': 'rep_ratio',
    'replay_start': 'rep_start', 'correction_term': 'corr_term', 'trust_region': 'treg',
    'delta': 'delta'
}

params = {
    # env params
    'policy': 'MlpPolicy',
    'seed': 0,
    'env_id': 'gridworld-v0',
    'singleAgent': False,
    'game_type': 'logic',  # Options: logic, contingency, switching_embodiments, logic_extended, contingency_extended, switching_embodiments_extended (mock self can be the real self), switching_embodiments_extended_1 (mock self cannot be the real self), switching_embodiments_extended_2 (harder). For switching_mappings, set 'shuffle_keys' to True and use 'contingency' as 'game_type'
    'player': 'option_critic',  # random, human, dqn_training, self_class, ppo2_training
    'exp_name': 'train_',
    'verbose': False,
    'n_levels': 100,
    'shuffle_keys': False, # Enable to play 'Switching Mappings' game

    # data params !add 'data_save_dir'
    'log_neptune': False,
    'data_save_dir': '../',
    'load': False,  # Load pretrained agent
    'timestamp': -1,  # Select which weight to run. Enter -1 to save only the last one.
    'save': True,  # Save the weights
    'levels_count': 1,  # Stop until 100 * 'levels_count' levels
    'load_game': None,  # Which weights to load
    'agent_location_random': True,  # Is agent location random or not
    'n_timesteps': 1000000000,
    'single_loc': False,
    'shuffle_each': 1,  # Shuffle each n levels
    'different_self_color': False,
    'load_str': '',  # Select which saved model to load
    'use_scratch_space': False,  # Save to scratch space
    'save_and_load_replay_buffer': False,
    'baselines_version': 2,
    'mid_modify': False,
    'modify_at': None,  # Modify the environment at 'modify_at' * 'n_levels' levels
    # Modify the environment to 'modify_to' after 2000 levels. Input levels_count as 40 to run for 4000 levels, if you set 'mid_modify' to True
    'modify_to': None,  # logic, contingency, switching_embodiments, logic_extended, contingency_extended
    # switching_embodiments_extended, switching_embodiments_extended_2
    'neg_reward': False,
    'n_cpu_tf_sess': 1,
    'keep_all_close': False,
    'switch_self_finding_100_lvls': False,  ## Switch to the harder perturbation every 50 levels
    'ten_r': False  ## Make mock self navigate only once in two levels
}

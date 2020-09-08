import os
from ray import tune

env_config = {
    "mass_home": os.environ["PWD"],
    "meta_file": "data/metadata_nomuscle.txt",
    "use_multi_env": False,
    "num_envs": 32,
}

common_config = {
    "env": "MyEnv",
    "env_config": {
        "mass_home": os.environ["PWD"],
        "meta_file": "data/metadata_nomuscle.txt",
        "use_multi_env": False,
        "num_envs": 32,
    },

    "framework": "torch",
    "extra_python_environs_for_driver": {
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
        "KMP_AFFINITY": "granularity=fine,compact,1,0"
    },
    "extra_python_environs_for_worker": {
        "OMP_NUM_THREADS": "20",
        "MKL_NUM_THREADS": "20",
        "KMP_AFFINITY": "granularity=fine,compact,1,0"
    },

    # "model": {
    #     "custom_model": "my_model",
    #     "custom_model_config": {},
    #     "max_seq_len": 0    # Placeholder value needed for ray to register model
    # },

    "model": {
        "fcnet_activation": "relu", # TODO: use LeakyReLU?
        "fcnet_hiddens": [256, 256],
        "vf_share_layers": False,
    },
}

ppo_config = common_config.update({
    "num_workers": 128,
    "num_cpus_per_worker": 1,
    "num_cpus_for_driver": 8,

    "use_critic": True,
    "use_gae": True,
    "lambda": 0.99,
    "gamma": 0.99,
    "kl_coeff": 0.2,
    "rollout_fragment_length": 64, # tune.grid_search([128, 256, 512]),
    "train_batch_size": 128*32,# tune.grid_search([128*32, 256*32, 512*32]),
    "sgd_minibatch_size": 128,
    "shuffle_sequences": True,
    "num_sgd_iter": 10,
    "lr": 1e-4,
    "lr_schedule": None,
    "vf_loss_coeff": 1.0,
    "entropy_coeff": 0.0,
    "entropy_coeff_schedule": None,
    "clip_param": 0.2,
    "vf_clip_param": 10.0,
    "grad_clip": None,
    "kl_target": 0.01,
    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "simple_optimizer": False,
})

ddppo_config = ppo_config.update({
    "num_workers": 4,
    "num_envs_per_worker": 8,
    "num_cpus_per_worker": 8,
})
del ddppo_config["train_batch_size"]

impala_config = common_config.update({
    "framework": "tf",

    "num_workers": 32,
    "num_gpus": 0,

    # "use_critic": True,
    # "use_gae": True,
    # "lambda": 0.99,
    "gamma": 0.99,
    # "clip_param": 0.2,
    # "kl_coeff": 0.2,
    "rollout_fragment_length": 128,
    "train_batch_size": 512,
    "min_iter_time_s": 10,
    "num_data_loader_buffers": 1,
    "minibatch_buffer_size": 1,
    "num_sgd_iter": 10,
    "replay_proportion": 0.0,
    "replay_buffer_num_slots": 100,
    "learner_queue_size": 16,
    "learner_queue_timeout": 300,
    "max_sample_requests_in_flight_per_worker": 2,
    "broadcast_interval": 1,
    "grad_clip": 40.0,
    "opt_type": "rmsprop",
    "lr": tune.loguniform(5e-6, 5e-3),
    "lr_schedule": None,
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": tune.grid_search([1e-1, 1e-3, 1e-5, 1e-7]),
    "vf_loss_coeff": 0.5,
    "entropy_coeff": tune.loguniform(5e-5, 1e-2),
    "entropy_coeff_schedule": None,
})

appo_config = impala_config.update({
    "num_workers": 160,
    "num_gpus": 0,

    "use_critic": True,
    "use_gae": True,
    "lambda": 0.99,
    "clip_param": 0.2,
    "kl_coeff": 0.2,
})

ars_config = common_config.update({
    "action_noise_std": 0.0,
    "noise_stdev": 0.005,  # std deviation of parameter noise
    "num_rollouts": 450,  # number of perturbs to try
    "rollouts_used": 270,  # number of perturbs to keep in gradient estimate
    "num_workers": 500,
    "sgd_stepsize": 0.025,  # sgd step-size
    "observation_filter": "MeanStdFilter",
    "noise_size": 250000000,
    "eval_prob": 0.03,  # probability of evaluating the parameter rewards
    "report_length": 10,  # how many of the last rewards we average over
    "offset": 0,
})
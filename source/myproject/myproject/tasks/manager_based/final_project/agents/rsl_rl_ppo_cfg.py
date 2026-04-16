from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.agents.rsl_rl_ppo_cfg import H1FlatPPORunnerCfg


@configclass
class FinalProjectUnitreeH1PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 6000
    clip_actions = 1.0
    save_interval = 100
    experiment_name = "final_project_unitree_h1_curriculum"
    run_name = ""
    resume = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        noise_std_type="log",
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.005,
        max_grad_norm=1.0,
    )


@configclass
class FinalProjectUnitreeH1MapPPORunnerCfg(FinalProjectUnitreeH1PPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 2500
    save_interval = 50
    experiment_name = "final_project_unitree_h1_map_finetune"
    run_name = ""
    resume = False

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class FinalProjectUnitreeH1BaselinePPORunnerCfg(H1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 3000
        self.clip_actions = 1.0
        self.save_interval = 50
        self.experiment_name = "final_project_unitree_h1_baseline"
        self.run_name = ""
        self.resume = False
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=0.25,
            noise_std_type="scalar",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
        )
        self.algorithm = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=5.0e-5,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.005,
            max_grad_norm=1.0,
        )

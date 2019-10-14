(import 'nl2code-0428-base.libsonnet')(output_from=false) {
    model+: {
        encoder+: {
            batch_encs_update: false,
        },
    },

    train+: {
        batch_size: 50, # this is overwrited
        max_steps: 10000, # since we're using meta_learning
        keep_every_n: 1000,
        eval_every_n: 100,
        save_every_n: 100,
        report_every_n: 1,
        enable_meta_learning: true,
    },

    lr_scheduler+: {
        start_lr: 1e-3, # since using meta data, lower the meta learning rate could be benefitial
    },

    meta_learning: {
        /*
        - almost all parameters in "train" section are used for meta part. 
            ("eval_batch_size" & "batch_size" are ignored)
        - "optimizer" and "lr_scheduler" section are also used for the meta part.
        */
        method: "other", # or "reptile", "other"
        internal_step: 1, # how many steps for the internal model update, 2 for maml, >= 2 for reptile
        internal_skip_first_step: false, # true for maml, false for default reptile

        meta_batch_size: 8, # aka "sampled task number per meta update", 1 for default reptile  
        update_batch_size: 16, # should be less than the instance number of the smallest task / or just use what we have
        update_learning_rate: 1e-4, # the learning rate for internal update, 10/7 changed to a more risky rate
        sample_task_distribution: "by_instance_num", # or "by_instance_num"

        enable_ft_evaluation: true, # only used for meta learning
        ft_batch_size: 32,
        ft_max_epoch: 8, # will report result for all
        ft_learning_rate: 1e-3, # need to double check
    },
}

(import 'nl2code-0428-base.libsonnet')(output_from=false) {
    model+: {
        encoder+: {
            batch_encs_update: false,
        },
    },

    train+: {
        batch_size: 50,
        enable_meta_learning: true,
    },

    meta_learning: {
        method: "MAML",
    },
}
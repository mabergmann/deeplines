import mlflow

lr_options = [1e-5]
size_options = [224]
weight_decay_options = [0.01, 0.001, 0.0001]
n_columns_options = [9]
backbones_options = ["pit_ti"]
batch_size_options = [16]
anchors_per_column_options = [5]
loss_options = ["hausdorff"]

loss_weights_options = [
    # [0.1, 0.1, 0.8],
    # [0.1, 0.3, 0.6],
    # [0.3, 0.1, 0.6],
    # [0.5, 0.1, 0.4],
    # [0.3, 0.3, 0.4],
    # [0.1, 0.5, 0.4],
    # [0.7, 0.1, 0.2],
    [0.5, 0.3, 0.2],
    # [0.3, 0.5, 0.2],
    # [0.01, 0.7, 0.29],
]
# loss_weights_options = [[0, .5, .5]]

for lr in lr_options:
    for size in size_options:
        for weight_decay in weight_decay_options:
            for n_columns in n_columns_options:
                for backbone in backbones_options:
                    for batch_size in batch_size_options:
                        for anchors_per_column in anchors_per_column_options:
                            for loss_weights in loss_weights_options:
                                for loss in loss_options:
                                    params = {
                                        "lr": lr,
                                        "width": size,
                                        "height": size,
                                        "weight_decay": weight_decay,
                                        "n_columns": n_columns,
                                        "backbone": backbone,
                                        "batch_size": batch_size,
                                        "anchors_per_column": anchors_per_column,
                                        "objectness_weight": loss_weights[0],
                                        "no_objectness_weight": loss_weights[1],
                                        "regression_weight": loss_weights[2],
                                        "dataset": "NKL",
                                        "loss": loss
                                    }

                                mlflow.projects.run(".", parameters=params, env_manager='local')

import tensorflow as tf
from Adventurers.model_generators.nogame_models import get_nodes_plan
from Adventurers.model_generators.no_game_metrics import NoGameSimplifiedLoss

def get_compile_args(
        learning_rate,
        node_incr_args,
        assignment_threshold,
        angles_num,
        label_smoothing,
        size,
        norm_ord,
        gamma,
        inf_value,
        hungarian
    ):

    _, _, turn_nodes = get_nodes_plan(**node_incr_args)

    compile_args = {
                'optimizer': tf.keras.optimizers.Adam(learning_rate),
                'loss': {'vertices': NoGameSimplifiedLoss(
                                                        turn_nodes=turn_nodes,
                                                        assignment_threshold=assignment_threshold,
                                                        angles_num=angles_num,
                                                        label_smoothing=label_smoothing,
                                                        size=size,
                                                        norm_ord=norm_ord,
                                                        gamma=gamma,
                                                        inf_value=inf_value,
                                                        hungarian=hungarian,
                                                        reduction='sum_over_batch_size',
                                                        name='NoGameLoss'
                                                        )},
            }
    
    return compile_args
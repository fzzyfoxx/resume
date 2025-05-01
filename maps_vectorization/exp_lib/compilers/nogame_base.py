import tensorflow as tf
from Adventurers.model_generators.no_game_metrics import AssignedNodeLoss, NewNode2VertexMinLoss, MaskedAssignedVertexLoss, NodeBCELoss
from Adventurers.model_generators.no_game_metrics import VerticesCoverageMetric, NodeF1Metric, NodePrecisionMetric, NodeRecallMetric, ProperNodesRatioMetric

def get_compile_args(
        learning_rate,
        norm_ord,
        label_smoothing,
        coverage_threshold,
        f1_threshold,
        loss_weights
    ):

    compile_args = {
        'optimizer': tf.keras.optimizers.Adam(learning_rate),
        'loss': {
            'SN_assig_loss': {'func': AssignedNodeLoss(norm_ord=norm_ord, name='SN_assig_loss'), 
                                        'inputs': ['active_node_assigned_vertex', 'shifted_nodes', 'squeezed_node_mask']
                                        },
            'NN_assig_loss': {'func': AssignedNodeLoss(norm_ord=norm_ord, name='NN_assig_loss'), 
                                    'inputs': ['node2vertex_assig', 'new_nodes', 'new_nodes_angle_mask']
                                    },
            'NN_min_loss': {'func': NewNode2VertexMinLoss(norm_ord=norm_ord, name='NN_min_loss'), 
                                'inputs': ['vertices_coords', 'new_nodes', 'new_nodes_vertices_mask', 'new_nodes_angle_mask']
                                },
            'V_loss': {'func': MaskedAssignedVertexLoss(norm_ord=norm_ord, name='V_loss'), 
                            'inputs': ['shifted_nodes', 'new_nodes', 'vertices_coords', 'all_nodes2vert_assig_mask', 'all_nodes2vert_reduced_mask']
                            },
            'SN_conf_loss': {'func': NodeBCELoss(label_smoothing=label_smoothing, name='SN_conf_loss'), 
                                    'inputs': ['shifted_nodes_conf_label', 'proper_conf', 'squeezed_node_mask']
                                    },
            'NN_conf_loss': {'func': NodeBCELoss(label_smoothing=label_smoothing, name='NN_conf_loss'), 
                                    'inputs': ['new_nodes_conf_label', 'new_nodes_conf', 'new_nodes_mask']
                                    },
        },
        'loss_weights': loss_weights,
        'metrics': {
            'vertices_coverage_metric': {'func': VerticesCoverageMetric(threshold=coverage_threshold, masked_dist=1e3, name='vertices_coverage_metric'), 
                                         'inputs': ['vertices_mask', 'point2vert_dists', 'active_node_assigned_vertex_mask']
                                         },
            'node_f1_metric': {'func': NodeF1Metric(threshold=f1_threshold, name='node_f1_metric'), 
                               'inputs': ['proper_conf', 'f1_weights', 'points2vert_dists_conf_label']
                               },
            'node_precision_metric': {'func': NodePrecisionMetric(threshold=f1_threshold, name='node_precision_metric'), 
                                      'inputs': ['proper_conf', 'f1_weights', 'points2vert_dists_conf_label']
                                      },
            'node_recall_metric': {'func': NodeRecallMetric(threshold=f1_threshold, name='node_recall_metric'), 
                                   'inputs': ['proper_conf', 'f1_weights', 'points2vert_dists_conf_label']
                                   },
            'proper_nodes_ratio_metric': {'func': ProperNodesRatioMetric(name='proper_nodes_ratio_metric'), 
                                          'inputs': ['points2vert_dists_conf_label', 'squeezed_node_mask']
                                          },
        }
            }
    
    return compile_args
import tensorflow as tf
from Adventurers.model_generators.no_game_metrics import AssignedNodeLoss, NewNode2VertexMinLoss, MaskedAssignedVertexLoss, NodeBCELoss
from Adventurers.model_generators.no_game_metrics import NodeEntropyLoss, NodeFocusLoss, TopKVertexLoss, VertexUncoveredAttractionLoss, WeightedVertexLoss
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
                #'shifted_node_assig_loss': {'func': AssignedNodeLoss(norm_ord=norm_ord, axis=-1, name='SN_assig_loss'), 
                #                            'inputs': ['active_node_assigned_vertex', 'shifted_nodes', 'squeezed_node_mask']},
                'assignment_entropy_loss': {'func': NodeEntropyLoss(gamma=0.6, norm_ord=norm_ord, name='assignment_entropy_loss'), 
                                            'inputs': ['shifted_nodes', 'sorted_vertices', 'vertices_topk_mask', 'squeezed_node_mask']},
                'new_node_assig_loss': {'func': AssignedNodeLoss(norm_ord=norm_ord, axis=[-1,-2], name='NN_assig_loss'), 
                                        'inputs': ['node2vertex_assig', 'new_nodes', 'new_nodes_angle_mask']},
                'new_node_min_loss': {'func': NewNode2VertexMinLoss(norm_ord=norm_ord, name='NN_min_loss'), 
                                      'inputs': ['vertices_coords', 'new_nodes', 'new_nodes_vertices_mask', 'new_nodes_angle_mask']},
                'node_focus_loss': {'func': NodeFocusLoss(gamma=0.1, norm_ord=norm_ord, name='node_focus_loss'), 
                                    'inputs': ['shifted_nodes', 'sorted_vertices', 'vertices_topk_mask', 'squeezed_node_mask']},
                'assig_vertex_loss': {'func': MaskedAssignedVertexLoss(norm_ord=norm_ord, name='vertex_loss'), 
                                      'inputs': ['shifted_nodes', 'new_nodes', 'vertices_coords', 'all_nodes2vert_assig_mask', 'all_nodes2vert_reduced_mask']},
                'topk_vertex_loss': {'func': TopKVertexLoss(norm_ord=norm_ord, name='vertex_loss'), 
                                     'inputs': ['shifted_nodes', 'sorted_vertices', 'vertices_topk_mask', 'squeezed_node_mask']},
                'vertex_uncovered_attr_loss': {'func': VertexUncoveredAttractionLoss(delta=4.55, norm_ord=norm_ord, name='vertex_uncovered_attr_loss'), 
                                               'inputs': ['shifted_nodes', 'sorted_vertices', 'vertices_topk_mask', 'squeezed_node_mask', 'sorted_vert2point_min_dist']},
                'vertex_loss': {'func': WeightedVertexLoss(beta=4.7, norm_ord=norm_ord, name='vertex_loss'), 
                                'inputs': ['shifted_nodes', 'sorted_vertices', 'vertices_topk_mask', 'squeezed_node_mask']},
                'shifted_node_conf_loss': {'func': NodeBCELoss(label_smoothing=label_smoothing, axis=-1, name='SN_conf_loss'), 
                                           'inputs': ['shifted_nodes_conf_label', 'proper_conf', 'squeezed_node_mask']},
                'new_nodes_conf_loss': {'func': NodeBCELoss(label_smoothing=label_smoothing, axis=[-1,-2], name='NN_conf_loss'), 
                                        'inputs': ['new_nodes_conf_label', 'new_nodes_conf', 'new_nodes_mask']},
            },
        'loss_weights': loss_weights,
        'metrics': {
            'vertices_coverage_metric': {'func': VerticesCoverageMetric(threshold=coverage_threshold, masked_dist=1e3, name='vertices_coverage_metric'), 
                                         'inputs': ['vertices_topk_mask_exp', 'point2vert_dists', 'active_node_assigned_vertex_mask']
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
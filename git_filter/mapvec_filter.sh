git filter-repo \
# Synthetic map generator
--path maps_vectorization/src/legend.py \
--path maps_vectorization/src/map_drawing.py \
--path maps_vectorization/src/map_generator.py \
--path maps_vectorization/src/pattern_generator.py \
--path maps_vectorization/src/patterns.py \
--path maps_vectorization/src/utils.py \
--path maps_vectorization/src/README.md \
--path maps_vectorization/src/map_args.json \
--path maps_vectorization/src/map_concatenation_args.json \
--path maps_vectorization/src/minimap_args.json \
--path maps_vectorization/src/parcel_input_args.json \
--path maps_vectorization/src/screenshots/ \
--path maps_vectorization/src/fonts/ \
# Simple pattern generator
--path maps_vectorization/models_src/VecDataset.py \
# RRPE
--path maps_vectorization/models_src/VecModels.py \
--path maps_vectorization/Fourier/pixel_features_model.ipynb \
--path maps_vectorization/Fourier/pixel_similarity_model.ipynb \
--path maps_vectorization/Fourier/radial_relative_pos_enc.ipynb \
--path maps_vectorization/Fourier/radial_enc_pixel_features.ipynb \
--path maps_vectorization/Fourier/radial_enc_vec_detection.ipynb \
--path maps_vectorization/Fourier/pixel_similarity_shapes_model.ipynb \
--path maps_vectorization/Fourier/model_generators/ \
# Experiment tracking library
--path maps_vectorization/exp_lib/
--path maps_vectorization/models_src/Trainer.py \
--path maps_vectorization/models_src/Trainer_support.py \
--path maps_vectorization/models_src/Support.py \
# Frequency-Domain Recognition of Linear Structures in Heavy Noise
--path maps_vectorization/Fourier/angle_shift.ipynb \
--path maps_vectorization/Fourier/lin_freq_adj.ipynb \
--path maps_vectorization/models_src/Hough.py \
--path maps_vectorization/models_src/fft_lib.py \
--path maps_vectorization/models_src/Attn_variations.py \
# Minor experiments
--path maps_vectorization/Vertex_detection/install_CRF.sh \
--path maps_vectorization/Vertex_detection/CRF_imports.py \
--path maps_vectorization/Vertex_detection/stages_flow.ipynb \
--path maps_vectorization/RPN_optimization.ipynb \
# Architectures
--path maps_vectorization/models_src/DETR.py \
--path maps_vectorization/models_src/Mask_RCNN.py \
--path maps_vectorization/models_src/UNet_model.py \
--path maps_vectorization/models_src/SegNet_model.py \
--path maps_vectorization/models_src/Kmeans.py \
--path maps_vectorization/models_src/backbones.py \
--path maps_vectorization/models_src/CombinedMetricsModel.py \
# Documentation
--path maps_vectorization/README.md \
--path maps_vectorization/RRPE.md \
# Other
--path maps_vectorization/models_src/Support.py \
--force \
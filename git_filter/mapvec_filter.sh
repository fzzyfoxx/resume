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
# RPN anchor optimization
--path maps_vectorization/RPN_optimization.ipynb \
# Minor experiments
--path maps_vectorization/Vertex_detection/install_CRF.sh \
--path maps_vectorization/Vertex_detection/CRF_imports.py \
--path maps_vectorization/Vertex_detection/stages_flow.ipynb \
# Architectures
--path maps_vectorization/models_src/DETR.py \
--path maps_vectorization/models_src/Mask_RCNN.py \
--path maps_vectorization/models_src/UNet_model.py \
--path maps_vectorization/models_src/SegNet_model.py \
--path maps_vectorization/models_src/Kmeans.py \
--path maps_vectorization/models_src/backbones.py \
# Documentation
--path maps_vectorization/README.md \
--force \
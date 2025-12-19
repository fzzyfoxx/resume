git clone https://github.com/sadeepj/crfasrnn_keras.git ./maps_vectorization/models_src/CRF

python3 maps_vectorization/Vertex_detection/CRF_imports.py

(cd maps_vectorization/models_src/CRF/src/cpp && make)


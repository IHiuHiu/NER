
�root"_tf_keras_model*�
{"name": "model_with_crf_loss_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "ModelWithCRFLoss", "config": {}, "shared_object_id": 0, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}, "is_graph_network": false, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50]}, "int32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50]}, "int32", "input_1"]}, "keras_version": "2.14.0", "backend": "tensorflow", "model_config": {"class_name": "ModelWithCRFLoss", "config": {}}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Custom>Adam", "config": {"name": "Adam", "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "jit_compile": false, "is_legacy_optimizer": false, "learning_rate": 0.0010000000474974513, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}}}}2
�Groot.base_model"_tf_keras_network*�G{"name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model_2", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "input_dim": 35179, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": 50}, "name": "embedding_2", "inbound_nodes": [[["input_5", 0, 0, {}]]]}, {"class_name": "Bidirectional", "config": {"name": "bidirectional_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}}, "merge_mode": "concat"}, "name": "bidirectional_2", "inbound_nodes": [[["embedding_2", 0, 0, {}]]]}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "name": "lstm_5", "inbound_nodes": [[["bidirectional_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["lstm_5", 0, 0, {}]]]}, {"class_name": "CustomLayer", "config": {"name": "custom_layer_2", "trainable": true, "dtype": "float32", "sublayer": {"class_name": "CRF", "config": {"name": "crf_2", "trainable": true, "dtype": "float32", "chain_initializer": "orthogonal"}}}, "name": "custom_layer_2", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_5", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["custom_layer_2", 0, 0], ["custom_layer_2", 0, 1], ["custom_layer_2", 0, 2], ["custom_layer_2", 0, 3]]}}, "shared_object_id": 20, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 50]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50]}, "float32", "input_5"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 50]}, "float32", "input_5"]}, "keras_version": "2.14.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}, "name": "input_5", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "input_dim": 35179, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": 50}, "name": "embedding_2", "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Bidirectional", "config": {"name": "bidirectional_2", "trainable": true, "dtype": "float32", "layer": {"class_name": "LSTM", "config": {"name": "lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 4}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 8}, "merge_mode": "concat"}, "name": "bidirectional_2", "inbound_nodes": [[["embedding_2", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "name": "lstm_5", "inbound_nodes": [[["bidirectional_2", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["lstm_5", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "CustomLayer", "config": {"name": "custom_layer_2", "trainable": true, "dtype": "float32", "sublayer": {"class_name": "CRF", "config": {"name": "crf_2", "trainable": true, "dtype": "float32", "chain_initializer": "orthogonal"}, "shared_object_id": 18}}, "name": "custom_layer_2", "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 19}], "input_layers": [["input_5", 0, 0]], "output_layers": {"class_name": "__tuple__", "items": [["custom_layer_2", 0, 0], ["custom_layer_2", 0, 1], ["custom_layer_2", 0, 2], ["custom_layer_2", 0, 3]]}}}}2
�	root.metrics_fn"_tf_keras_metric*�{"class_name": "Accuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32"}, "shared_object_id": 22}2
�
root.loss_tracker"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 23}2
�(root.base_model.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_5", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_5"}}2
�)$root.base_model.layer_with_weights-0"_tf_keras_layer*�{"name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Embedding", "config": {"name": "embedding_2", "trainable": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 50]}, "input_dim": 35179, "output_dim": 100, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": null, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": true, "input_length": 50}, "inbound_nodes": [[["input_5", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50]}}2
�
�+$root.base_model.layer_with_weights-2"_tf_keras_rnn_layer*�{"name": "lstm_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTM", "config": {"name": "lstm_5", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "inbound_nodes": [[["bidirectional_2", 0, 0, {}]]], "shared_object_id": 14, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 200]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 200]}}2
�,$root.base_model.layer_with_weights-3"_tf_keras_layer*�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 18, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["lstm_5", 0, 0, {}]]], "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 200]}}2
�-$root.base_model.layer_with_weights-4"_tf_keras_layer*�{"name": "custom_layer_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CustomLayer", "config": {"name": "custom_layer_2", "trainable": true, "dtype": "float32", "sublayer": {"class_name": "CRF", "config": {"name": "crf_2", "trainable": true, "dtype": "float32", "chain_initializer": "orthogonal"}, "shared_object_id": 18}}, "inbound_nodes": [[["dense_2", 0, 0, {}]]], "shared_object_id": 19, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 18]}}2
�M2root.base_model.layer_with_weights-1.forward_layer"_tf_keras_rnn_layer*�{"name": "forward_lstm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTM", "config": {"name": "forward_lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "zero_output_for_mask": true, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 27}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 31, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 32}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 100]}}2
�N3root.base_model.layer_with_weights-1.backward_layer"_tf_keras_rnn_layer*�{"name": "backward_lstm_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTM", "config": {"name": "backward_lstm_4", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": true, "stateful": false, "unroll": false, "time_major": false, "zero_output_for_mask": true, "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 37, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 100]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 38}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 100]}}2
�V)root.base_model.layer_with_weights-2.cell"_tf_keras_layer*�{"name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 10}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 13, "build_input_shape": {"class_name": "__tuple__", "items": [null, 200]}}2
�d-root.base_model.layer_with_weights-4.sublayer"_tf_keras_layer*�{"name": "crf_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "CRF", "config": {"name": "crf_2", "trainable": true, "dtype": "float32", "chain_initializer": "orthogonal"}, "shared_object_id": 18, "build_input_shape": {"class_name": "TensorShape", "items": [null, 50, 18]}}2
�	�7root.base_model.layer_with_weights-1.forward_layer.cell"_tf_keras_layer*�{"name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 27}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 30, "build_input_shape": {"class_name": "__tuple__", "items": [null, 100]}}2
�	�8root.base_model.layer_with_weights-1.backward_layer.cell"_tf_keras_layer*�{"name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "LSTMCell", "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 100, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.5, "recurrent_dropout": 0.5, "implementation": 1}, "shared_object_id": 36, "build_input_shape": {"class_name": "__tuple__", "items": [null, 100]}}2
��9root.base_model.layer_with_weights-4.sublayer.accuracy_fn"_tf_keras_metric*�{"class_name": "Accuracy", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32"}, "shared_object_id": 39}2
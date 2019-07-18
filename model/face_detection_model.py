import numpy as np
import os
import tensorflow as tf
from sklearn.utils import shuffle

from util import Config, ImgUtil

class FaceDetectionModel:
    """
    TensorFlow model for classifying the mood for hand-drawn faces
    """

    learning_rate = 0.001  # Learning rate for the model
    dropout_rate = 0.4  # Rate at which weights are turned off to prevent over-fitting
    mood_features = {  # The mood features in the dataset
        'happy': 0,
        'meh': 1,
        'sad': 2
    }

    def __init__(self, config):
        """

        :type config: Config
        """
        self.config = config.face_detection
        self.dataset = {}

        # Create the estimator
        self.classifier = tf.estimator.Estimator(model_fn=self._conv_net_model,
                                                 model_dir=self.config['checkpoint_dir'])
        self.tensors_to_log = {"probabilities": "softmax_tensor"}
        self.logging_hook = tf.train.LoggingTensorHook(tensors=self.tensors_to_log, every_n_iter=10)

    def train(self, steps=100):
        """

        :type steps: int
        :param steps: int
        :return:
        """
        x_train, y_train = self._get_features_labels_tuple(self.config['train_data_dir'])
        assert len(x_train) == len(y_train)
        assert type(x_train) == np.ndarray
        assert type(y_train) == np.ndarray

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_train},
            y=y_train,
            batch_size=40,
            num_epochs=None,
            shuffle=True)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=steps,
            hooks=[self.logging_hook])

    def evaluate(self):
        """

        :return:
        """
        x_test, y_test = self._get_features_labels_tuple(self.config['testing_data_dir'])

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": x_test},
            y=y_test,
            num_epochs=1,
            shuffle=False)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    def predict(self, img_vec):
        """

        :return:
        """
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(img_vec)},
            num_epochs=1,
            shuffle=False)
        prediction_results = self.classifier.predict(predict_input_fn)
        for p in prediction_results:
            return p

    def _get_features_labels_tuple(self, path, enable_shuffle=True):
        """
        
        :type shuffle: bool
        :type path: str
        :return: tuple (features, labels)
        """
        w, h = 36, 36
        dataset_dict = {k: ImgUtil.get_batch_image_vectors(f'{path}/{k}', w, h)
                        for k in FaceDetectionModel.mood_features.keys()}
        x = np.concatenate([v for k, v in dataset_dict.items()])
        y = np.concatenate([[FaceDetectionModel.mood_features[k]] * len(v) for k, v in dataset_dict.items()])
        z = list(zip(x, y))
        if enable_shuffle:
            z = shuffle(z)
        x, y = zip(*z)
        assert len(x) == len(y)
        return np.array(x), np.array(y)

    @staticmethod
    def _conv_net_model(features, labels, mode):

        """
        Builds the conv-net model with the given features and labels.

        :param features: dict of features with key 'x'
        :param labels: list of labels corresponding to the features
        :param mode: training mode. see tf.estimator.ModeKeys
        :return: EstimatorSpec
        """
        input_layer = tf.reshape(features['x'], [-1, 36, 36, 1])
        # Kernel size
        k_size = [3, 3]

        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=k_size,
                                 padding='same', activation=tf.nn.relu)

        # Pool 1 (36, 36) -> (18, 18)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=k_size, padding='same',
                                 activation=tf.nn.relu)

        # Pool 2 (18, 18) -> (9, 9)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Flat layer
        pool2_flat = tf.reshape(pool2, [-1, 9 * 9 * 64])

        # Fully connected dense layer
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        # Dropout
        dropout = tf.layers.dropout(inputs=dense, rate=FaceDetectionModel.dropout_rate,
                                    training=(mode == tf.estimator.ModeKeys.TRAIN))

        # Logits
        logits = tf.layers.dense(inputs=dropout, units=len(FaceDetectionModel.mood_features), activation=None)
        predictions = {
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }

        # Handle predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Compute loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        # Perform gradient descent when training
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=FaceDetectionModel.learning_rate)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for validation)
        eval_metrics = {
            "accuracy": tf.metrics.accuracy(labels=labels,
                                            predictions=predictions["classes"],
                                            name="accuracy")
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)


if __name__ == '__main__':
    config_yml_path = os.path.dirname(os.path.realpath(__file__)) + '/config/config.yml'
    print('config yml path', config_yml_path)
    config = Config(config_yml_path)
    model = FaceDetectionModel(config)
    model.evaluate()



import numpy as np
import tensorflow as tf


class PolicyModel:
    def __init__(self, input_dim, n_actions, feature_transformer=None, epsilon=0.1, lr=0.001):
        self.epsilon = epsilon
        self.feature_transformer = feature_transformer
        self.n_actions = n_actions
        self.model = self._create_model(input_dim, n_actions)
        self.optimizer = tf.keras.optimizers.Adam(lr)

    def _create_model(self, input_dim, n_actions):
        # Define input layers
        state_input = tf.keras.layers.Input(shape=(input_dim,), name='state', dtype=tf.float32)

        # Define dense layers
        x = tf.keras.layers.Dense(32, activation='relu')(state_input)
        x = tf.keras.layers.Dense(32, activation='relu')(x)

        # Output layer
        output = tf.keras.layers.Dense(n_actions, activation='softmax')(x)

        # Construct and compile the model
        model = tf.keras.models.Model(inputs=[state_input], outputs=[output])

        return model

    @staticmethod
    def custom_loss(p_a_given_s, advantages, actions):
        selected_probs = tf.math.log(tf.reduce_sum(p_a_given_s * actions))
        return -tf.reduce_sum(selected_probs * advantages)

    @tf.function
    def train_step(self, X, advantages, actions):
        with tf.GradientTape() as tape:
            p_a_given_s = self.model(X, training=True)
            loss = self.custom_loss(p_a_given_s, advantages, actions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def partial_fit(self, X, actions, advantages):
        X = self.get_input(X)
        # Convert actions and advantages to correct format
        actions = tf.convert_to_tensor(np.atleast_2d(actions), dtype=tf.float32)
        advantages = tf.convert_to_tensor(np.atleast_2d(advantages), dtype=tf.float32)
        # Create 'one-hot' encoded version of actions to use with the custom loss
        action_one_hot = tf.cast(tf.keras.utils.to_categorical(actions, self.n_actions), dtype=tf.float32)
        loss = self.train_step(X, advantages, action_one_hot)
        return loss

    def get_input(self, X):
        if self.feature_transformer is None:
            return np.atleast_2d(X)
        else:
            return self.feature_transformer.transform(np.atleast_2d(X))

    def predict(self, X):
        X = self.get_input(X)
        # dummy_advantages = np.zeros(shape=(len(X), 1))
        # dummy_actions = np.zeros(shape=(len(X), 1))
        prob = self.model.predict(X, verbose=0)
        return prob

    def sample_action(self, X, greedy=False):
        probs = self.predict(X)

        if greedy:
            return np.argmax(probs[0])
        else:
            return np.random.choice(np.arange(self.n_actions), p=probs[0])
        # if np.random.random() < self.epsilon:
        #     return np.random.choice(range(self.n_actions))
        # else:
        #     probabilities = self.predict(X)
        #     return np.argmax(probabilities)


class ValueModel:
    def __init__(self, input_dim, feature_transformer):
        self.feature_transformer = feature_transformer
        self.model = self._create_model(input_dim)

    def _create_model(self, input_dim):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)  # single output unit for value estimation
        ])
        model.compile(optimizer='adam', loss='mse')  # Mean Squared Error is a common choice for regression tasks
        return model

    def partial_fit(self, X, y):
        X = self.feature_transformer.transform(np.atleast_2d(X))
        y = np.atleast_1d(y)
        self.model.train_on_batch(X, y)

    def predict(self, X):
        X = self.feature_transformer.transform(np.atleast_2d(X))
        return self.model.predict(X, verbose=0)[0][0]

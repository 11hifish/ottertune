import gc
import tensorflow as tf
import numpy as np

from util import get_analysis_logger

LOG = get_analysis_logger(__name__)


class GP_UCB(object):

    def __init__(self, length_scale=1.0, magnitude=1.0, max_train_size=7000,
                 batch_size=3000, num_threads=4, check_numerics=True, debug=False,
                 ucb_mean_mult=1.0, ucb_sigma_mult=3.0,
                 max_iter=100, learning_rate=0.01, epsilon=1e-6):
        assert np.isscalar(length_scale)
        assert np.isscalar(magnitude)
        assert length_scale > 0 and magnitude > 0
        # kernel constants
        self.length_scale = length_scale
        self.magnitude = magnitude
        # training constants
        self.max_train_size_ = max_train_size
        self.batch_size_ = batch_size
        self.num_threads_ = num_threads
        self.check_numerics = check_numerics
        self.debug = debug
        # UCB constants
        self.ucb_mean_multiplier = ucb_mean_mult
        self.ucb_sigma_multiplier = ucb_sigma_mult
        # UCB computation constants
        self.max_iter = max_iter
        # Constants for computing argmin UCB(x)
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        # computation inputs
        self.X_train_place = None
        self.ridge = None
        self.y_train_place = None
        self.X_test_place = None
        # important parameters
        self.K = None
        self.K_y = None
        self.X_argmin = None
        self.ucb = None
        # important computations
        self.calc_kernel = None
        self.calc_K_y = None
        self.gp_post_mean = None
        self.gp_post_sigma = None
        self.find_min_ucb = None
        self.assign_X_argmin = None
        self.store_X_train = None
        # construct computational graph
        self.graph = tf.Graph()
        # create a live session
        self.sess = tf.Session(graph=self.graph,
                               config=tf.ConfigProto(
                                   intra_op_parallelism_threads=self.num_threads_))
        with self.graph.as_default():
            self.build_graph_basic_gp()
            self.saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

    def _calc_euc_distance(self, X1, X2):
        """
            An elegant way of computing matrix Euclidean distances.
            source :
              https://gist.github.com/mbsariyildiz/34cdc26afb630e8cae079048eef91865
        """
        # squared norms of each row in A and B
        na = tf.reduce_sum(tf.square(X1), 1)
        nb = tf.reduce_sum(tf.square(X2), 1)
        # na as a row and nb as a co"lumn vectors
        na = tf.reshape(na, [-1, 1])
        nb = tf.reshape(nb, [1, -1])
        # return pairwise euclidead difference matrix
        res = tf.sqrt(tf.maximum(na - 2 * tf.matmul(X1, X2, False, True) + nb, 0.0))
        return res

    def _calc_exp_kernel(self, X1, X2, magnitude, length_scale, ridge=None):
        dist = self._calc_euc_distance(X1, X2)
        pure_exp_kernel = magnitude * tf.exp(-dist / length_scale)
        if ridge is None:
            return pure_exp_kernel
        else:
            return pure_exp_kernel + tf.diag(ridge)

    def build_graph_basic_gp(self):
        with tf.variable_scope('graph_gp'):
            # Nodes for distance computation
            self.X_train_place = tf.placeholder(tf.float64, name="X_train")
            self.X_train = tf.Variable([0], name='X_train_var',
                                       dtype=tf.float64, validate_shape=False)
            self.store_X_train = tf.assign(self.X_train, self.X_train_place,
                                           validate_shape=False)
            # Compute exp kernel
            self.K = tf.Variable([0], name='kernel', dtype=tf.float64, validate_shape=False)
            self.ridge = tf.placeholder(name='ridge', dtype=tf.float64)

            self.calc_kernel = tf.assign(self.K, self._calc_exp_kernel(self.X_train, self.X_train, self.magnitude,
                                         self.length_scale, self.ridge),
                                         validate_shape=False)
            # Compute kernel * y_train
            self.K_y = tf.Variable([0], name='K_y', dtype=tf.float64, validate_shape=False)
            self.y_train_place = tf.placeholder(name='y_train', dtype=tf.float64)
            K_inv = tf.matrix_inverse(self.K)
            self.calc_K_y = tf.assign(self.K_y, tf.matmul(K_inv, self.y_train_place),
                                      validate_shape=False)

            self.dist = self._calc_euc_distance(self.X_train, self.X_train)

    def build_graph_ucb(self, x_train_dim):
        with tf.variable_scope('graph_ucb'):
            # Input for prediction ( finding argmin UCB(x) )
            init_val = np.reshape(np.array([0] * x_train_dim), (1, -1))
            self.X_argmin = tf.Variable(init_val, name='X_argmin', dtype=tf.float64,
                                        validate_shape=False)
            self.X_test_place = tf.placeholder(name='X_test', dtype=tf.float64)
            self.assign_X_argmin = tf.assign(self.X_argmin, self.X_test_place,
                                             validate_shape=False)
            # K2 = Kernel(X_train, X_test)
            K2 = self._calc_exp_kernel(self.X_train, self.X_argmin,
                                       self.magnitude, self.length_scale)
            # GP posterior mean
            self.gp_post_mean = tf.cast(tf.matmul(tf.transpose(K2),
                                                  self.K_y),
                                        tf.float64)
            # GP posterior std
            K_inv = tf.matrix_inverse(self.K)
            self.gp_post_sigma = tf.cast((tf.sqrt(
                tf.diag_part(self.magnitude - tf.matmul(tf.transpose(K2),
                                                        tf.matmul(K_inv, K2))))),
                tf.float64)
            # Output for prediction
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                               epsilon=self.epsilon)
            self.ucb = tf.squeeze(tf.subtract(self.ucb_mean_multiplier * self.gp_post_mean,
                                              self.ucb_sigma_multiplier * self.gp_post_sigma))

            self.find_min_ucb = optimizer.minimize(tf.expand_dims(self.ucb, 0),
                                                   var_list=[self.X_argmin])

    def fit(self, X_train, y_train, ridge=0.1):
        X_train, y_train = self.check_X_y(X_train, y_train)

        sample_size = X_train.shape[0]
        X_train_dim = len(X_train[0])
        if np.isscalar(ridge):
            ridge = np.ones(sample_size) * ridge

        with self.graph.as_default():
            self.build_graph_ucb(X_train_dim)
            self.sess.run(tf.global_variables_initializer())

        ridge = np.ones(len(X_train)) * ridge
        # store X_train, for later computation
        self.sess.run(self.store_X_train, feed_dict={self.X_train_place: X_train})
        # compute kernel
        self.sess.run(self.calc_kernel, feed_dict={self.ridge: ridge})
        K_y_val = self.sess.run(self.calc_K_y, feed_dict={self.y_train_place: y_train})

        LOG.info('K val {}'.format(self.sess.run(self.K)))
        LOG.info('K * y {}'.format(K_y_val))
        LOG.info('X train {}'.format(self.sess.run(self.X_train)))
        return self

    def predict(self, X_test, X_min=None, X_max=None, constraint_helper=None,
                categorical_feature_method='hillclimbing',
                categorical_feature_steps=3):
        if not self._check_fitted():
            raise Exception("The model must be trained before making predictions!")

        X_test = np.float32(self.check_array(X_test))
        global_argmin_x = None
        global_min_ucb = None
        for x_idx in range(len(X_test)):
            x_t = X_test[x_idx]  # shape [1, None]
            x_t = np.reshape(x_t, (1, -1))
            self.sess.run(self.assign_X_argmin,
                          feed_dict={self.X_test_place: x_t})
            argmin_x = x_t
            ucb_x = None
            for step in range(self.max_iter):
                self.sess.run(self.find_min_ucb)
                ucb_val = self.sess.run(self.ucb)
                # constraint potential argmin UCB(x)
                argmin_valid = self.sess.run(self.X_argmin)
                if X_max is not None:
                    argmin_valid = np.minimum(argmin_valid, X_max)
                if X_min is not None:
                    argmin_valid = np.maximum(argmin_valid, X_min)
                self.sess.run(self.assign_X_argmin,
                              feed_dict={self.X_test_place: argmin_valid})
                if ucb_x is None or ucb_x > ucb_val:
                    argmin_x = argmin_valid
                    ucb_x = ucb_val
            LOG.info('min ucb now {}'.format(ucb_x))
            # update global
            if (global_min_ucb is None or global_min_ucb > ucb_x) and np.isfinite(ucb_x):
                global_min_ucb = ucb_x
                global_argmin_x = argmin_x
        return global_argmin_x, global_min_ucb

    def save_model(self, path='saved_gp/model.ckpt'):
        self.saver.save(self.sess, path)

    def restore_existing_model(self, path='saved_gp/model.ckpt'):
        self.saver.restore(self.sess, path)
        if self.X_train is None:
            raise Exception('Restore failed. No X train tensor.')
        X_train_val = self.sess.run(self.X_train)
        x_dim = X_train_val.shape[1]
        with self.graph.as_default():
            self.build_graph_ucb(x_dim)
            all_vars_graph_ucb = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                   scope='graph_ucb')
            self.sess.run(tf.initialize_variables(all_vars_graph_ucb))

    def check_X_y(self, X, y):
        from sklearn.utils.validation import check_X_y
        if X.shape[0] > self.max_train_size_:
            raise Exception("X_train size cannot exceed {} ({})"
                            .format(self.max_train_size_, X.shape[0]))
        return check_X_y(X, y, multi_output=True,
                         allow_nd=True, y_numeric=True,
                         estimator="GPR")

    # checking helpers
    def _check_fitted(self):
        return self.K is not None and self.K_y is not None and self.sess is not None

    @staticmethod
    def check_array(X):
        from sklearn.utils.validation import check_array
        return check_array(X, allow_nd=True, estimator="GPR")
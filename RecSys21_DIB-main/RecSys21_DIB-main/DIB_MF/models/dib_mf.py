import sys
sys.path.append("/home/wyf/RecSys21_DIB-main/RecSys21_DIB-main/DIB_MF")
import numpy as np
from tqdm import tqdm
import tensorflow as tf
import optuna
from optuna.samplers import TPESampler
from optuna.trial import Trial
from scipy.sparse import csr_matrix
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer
from scipy.sparse import vstack, lil_matrix
from utils import io
import numpy as np

class Objective:

    def __init__(self, num_users, num_items, optimizer, gpu_on, train, _train, valid, test, iters, metric, is_topK,
                 topK, seed) -> None:
        """Initialize Class"""
        self.num_users = num_users
        self.num_items = num_items
        self.optimizer = optimizer
        self.gpu_on = gpu_on
        self.train = train
        self._train = _train
        self.valid = valid
        self.test = test
        self.iters = iters
        self.metric = metric
        self.is_topK = is_topK
        self.topK = topK
        self.seed = seed

    def __call__(self, trial: Trial) -> float:
        """Calculate an objective value."""

        # sample a set of hyperparameters.
        rank = trial.suggest_discrete_uniform('rank', 5, 100, 5)
        lam = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048, 4096, 8192, 16384])
        lr = trial.suggest_categorical('learning_rate', [0.001, 0.005, 0.01, 0.05, 0.1])
        alpha = trial.suggest_uniform('alpha', 0.1, 0.2)
        gamma = trial.suggest_uniform('gamma', 0.01, 0.1)

        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)#tf.set_random_seed(self.seed)

        model = DIBMF(self.num_users, self.num_items, np.int(rank), np.int(batch_size), lamb=lam, alpha=alpha,
                      gamma=gamma, learning_rate=lr, optimizer=self.optimizer, gpu_on=self.gpu_on)

        score, _, _, _, _, _ = model.train_model(self.train, self._train, self.valid, self.test, self.iters,
                                                 self.metric, self.topK, self.is_topK, self.gpu_on, self.seed)

        model.sess.close()
        tf.compat.v1.reset_default_graph()

        return score


class Tuner:
    """Class for tuning hyperparameter of MF models."""
    
    def __init__(self):
        """Initialize Class."""

    def tune(self, n_trials, num_users, num_items, optimizer, gpu_on, train, _train, valid, test, epoch, metric, topK,
             is_topK, seed):
        """Hyperparameter Tuning by TPE."""
        # 定义一个目标函数，用于优化超参数
        objective = Objective(num_users=num_users, num_items=num_items, optimizer=optimizer, gpu_on=gpu_on, train=train,
                              _train=_train, valid=valid, test=test, iters=epoch, metric=metric, is_topK=is_topK,
                              topK=topK, seed=seed)
        # 创建一个使用TPE算法的Optuna学习实例
        study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
        # 在指定的试验次数内运行超参数优化
        study.optimize(objective, n_trials=n_trials)
        # 返回最优超参数组合        
        return study.trials_dataframe(), study.best_params
        

class DIBMF(object):
    def __init__(self, num_users, num_items, embed_dim, batch_size,
                 lamb=0.01,
                 alpha=0.01,
                 gamma=0.01,
                 learning_rate=1e-3,
                 optimizer=tf.compat.v1.train.AdamOptimizer,
                 gpu_on=False,
                 **unused):
        self._num_users = num_users
        self._num_items = num_items
        self._embed_dim = embed_dim
        self._lamb = lamb
        self._alpha = alpha
        self._gamma = gamma
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._gpu_on = gpu_on
        self._build_graph()
    #初始化
    def _build_graph(self):

        with tf.compat.v1.variable_scope('dib-mf'):# 声明变量作用域
            # 声明输入数据的Placeholder
            tf.compat.v1.disable_eager_execution()
            self.user_idx = tf.compat.v1.placeholder(tf.compat.v1.int32, [None])#tf.placeholder
            self.item_idx = tf.compat.v1.placeholder(tf.compat.v1.int32, [None])#qudiao compat.v1
            self.label = tf.compat.v1.placeholder(tf.compat.v1.float32, [None])

            # 定义需要学习的参数Variable
            z_user_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal([self._num_users, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            c_user_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal([self._num_users, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            # 定义不需要学习的、全0的Variable
            user_zero_vector = tf.compat.v1.get_variable(
                'user_zero_vector', [self._num_users, self._embed_dim],
                initializer=tf.compat.v1.constant_initializer(0.0, dtype=tf.float32), trainable=False)
            # 将全0的Variable与需要学习的Variable拼接在一起
            self.z_user_embeddings = tf.compat.v1.concat([z_user_embeddings, user_zero_vector], 1)
            self.c_user_embeddings = tf.compat.v1.concat([user_zero_vector, c_user_embeddings], 1)

            z_item_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal([self._num_items, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            c_item_embeddings = tf.compat.v1.Variable(tf.compat.v1.random_normal([self._num_items, self._embed_dim],
                                                             stddev=1 / (self._embed_dim ** 0.5), dtype=tf.float32))
            item_zero_vector = tf.compat.v1.get_variable(
                'item_zero_vector', [self._num_items, self._embed_dim],
                initializer=tf.compat.v1.constant_initializer(0.0, dtype=tf.float32), trainable=False)

            self.z_item_embeddings = tf.compat.v1.concat([z_item_embeddings, item_zero_vector], 1)
            self.c_item_embeddings = tf.compat.v1.concat([item_zero_vector, c_item_embeddings], 1)

            

            with tf.compat.v1.variable_scope("mf_loss"):# 声明变量作用域
                z_users = tf.nn.embedding_lookup(self.z_user_embeddings, self.user_idx)
                z_items = tf.nn.embedding_lookup(self.z_item_embeddings, self.item_idx)
                z_x_ij = tf.reduce_sum(tf.multiply(z_users, z_items), axis=1)
                #从嵌入矩阵中查找用户嵌入向量和物品嵌入向量，并进行逐元素相乘和求和，得到评分预测值。
                c_users = tf.nn.embedding_lookup(self.c_user_embeddings, self.user_idx)
                c_items = tf.nn.embedding_lookup(self.c_item_embeddings, self.item_idx)
                c_x_ij = tf.reduce_sum(tf.multiply(c_users, c_items), axis=1)

                zc_users = z_users + c_users
                zc_items = z_items + c_items
                zc_x_ij = tf.reduce_sum(tf.multiply(zc_users, zc_items), axis=1)

                mf_loss = tf.reduce_mean(
                    (1 - self._alpha) * tf.nn.sigmoid_cross_entropy_with_logits(logits=z_x_ij, labels=self.label) -
                    self._gamma * tf.nn.sigmoid_cross_entropy_with_logits(logits=c_x_ij, labels=self.label) +
                    self._alpha * tf.nn.sigmoid_cross_entropy_with_logits(logits=zc_x_ij, labels=self.label))

                self.a = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=z_x_ij, labels=self.label))
                self.b = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=c_x_ij, labels=self.label))
                self.c = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=zc_x_ij, labels=self.label))

            with tf.compat.v1.variable_scope('l2_loss'):
                unique_user_idx, _ = tf.unique(self.user_idx)
                unique_users = tf.nn.embedding_lookup(self.z_user_embeddings, unique_user_idx)

                unique_item_idx, _ = tf.unique(self.item_idx)
                unique_items = tf.nn.embedding_lookup(self.z_item_embeddings, unique_item_idx)

                l2_loss = tf.reduce_mean(tf.nn.l2_loss(unique_users)) + tf.reduce_mean(tf.nn.l2_loss(unique_items))

            with tf.compat.v1.variable_scope('loss'):
                self._loss = mf_loss + self._lamb * l2_loss

            with tf.compat.v1.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)

            with tf.compat.v1.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)#,yihou

            if self._gpu_on:
                config = tf.compat.v1.ConfigProto()
                config.gpu_options.allow_growth = True
            else:
                config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
            self.sess = tf.compat.v1.Session(config=config)
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)

    @staticmethod
    #生成一个二维数组，其中每个元素都是一个由两个整数组成的元组，表示用户和物品的索引
    def generate_total_sample(num_users, num_items):
        r0 = np.arange(num_users)
        r1 = np.arange(num_items)

        out = np.empty((num_users, num_items, 2), dtype=np.dtype(np.int32))
        out[:, :, 0] = r0[:, None]
        out[:, :, 1] = r1

        return out.reshape(-1, 2)

    @staticmethod
    #生成一个二维数组，其中每个元素都是一个由两个整数组成的元组，表示用户和物品的索引
    def get_batches(pos_ui, num_pos, unlabeled_ui, num_unlabeled, batch_size):
        # 生成一个随机的索引数组，用于从正样本中抽取batch_size个样本
        pos_idx = np.random.choice(np.arange(num_pos), size=np.int(batch_size))
        # 生成一个随机的索引数组，用于从负样本中抽取1.5*batch_size个样本
        unlabeled_idx = np.random.choice(np.arange(num_unlabeled), size=np.int(1.5 * batch_size))
        # 将正样本数据的随机索引和未标记样本数据的随机索引合并为一个训练集。
        train_batch = np.r_[pos_ui[pos_idx], unlabeled_ui[unlabeled_idx]]
        # 训练集的标签为训练集的第3列数据，即用户和商品的交互标记
        train_label = train_batch[:, 2]
        #返回训练集的数据和标签。
        return train_batch, train_label
    
    def train_model(self, matrix_train, _matrix_train, matrix_valid, matrix_test, epoch=100, metric='AUC', topK=50,
                    is_topK=False, gpu_on=False, seed=0):
        # 设置随机数种子，保证结果可重复性
        np.random.seed(seed)
        tf.compat.v1.set_random_seed(seed)
        # 将训练数据转换成稀疏矩阵
        user_item_matrix = lil_matrix(matrix_train)
         # 获取稀疏矩阵中非零元素的位置信息（即用户-物品配对信息）
        user_item_pairs = np.asarray(user_item_matrix.nonzero(), order='F').T
        # 获取所有用户和物品的配对信息
        all_ui_pair = self.generate_total_sample(self._num_users, self._num_items)
        # 将用户-物品配对信息的行转换为元组
        user_item_pairs_rows = user_item_pairs.view([('', user_item_pairs.dtype)] * user_item_pairs.shape[1])
        # 将所有用户-物品配对信息的行转换为元组
        all_ui_pair_rows = all_ui_pair.view([('', all_ui_pair.dtype)] * all_ui_pair.shape[1])
        # 通过差集操作获取所有未标记的用户-物品配对信息
        unlabeled_ui_pair = np.setdiff1d(
            all_ui_pair_rows, user_item_pairs_rows).view(all_ui_pair.dtype).reshape(-1, all_ui_pair.shape[1])
        # 将用户-物品配对信息和未标记的用户-物品配对信息合并为一个训练集，并将正样本和未标记样本的标签加到数组的第三列中
        train_ui = np.r_[np.c_[user_item_pairs, np.ones(user_item_pairs.shape[0])],
                         np.c_[unlabeled_ui_pair, np.zeros(unlabeled_ui_pair.shape[0])]].astype('int32')
        # 获取正样本和未标记样本的数量
        pos_train = train_ui[train_ui[:, 2] == 1]
        # 获取正样本的数量
        num_pos = np.sum(train_ui[:, 2].astype(np.int))
        # 提取未标记的用户-物品配对信息
        unlabeled_train = train_ui[train_ui[:, 2] == 0]
        # 获取未标记样本的数量
        num_unlabeled = np.sum(1 - train_ui[:, 2])

        # Training
        best_result, best_RQ, best_X, best_xBias, best_Y, best_yBias = 0, None, None, None, None, None
        result_early_stop = 0
        # 通过tqdm库显示训练进度
        for i in tqdm(range(epoch)):
            # 获取训练集的数据和标签，每次训练时从正样本和未标记样本中抽取batch_size个样本
            train_batch, train_label = self.get_batches(pos_train, num_pos, unlabeled_train, num_unlabeled,
                                                        self._batch_size)
            # 将训练集的数据和标签输入到模型中进行训练，同时获取模型的损失函数值
            feed_dict = {self.user_idx: train_batch[:, 0],
                         self.item_idx: train_batch[:, 1],
                         self.label: train_label,
                         }
            # 运行模型，同时获取模型的损失函数值
            _, a, b, c = self.sess.run([self._train, self.a, self.b, self.c], feed_dict=feed_dict)
            # 获取当前模型的隐向量矩阵
            RQ, Y = self.sess.run([self.z_user_embeddings, self.z_item_embeddings])
            # 对测试集进行评估，计算评估指标
            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         topK=topK,
                                                         matrix_Train=_matrix_train,
                                                         matrix_Test=matrix_valid,
                                                         is_topK=is_topK,
                                                         gpu=gpu_on)
            valid_result = evaluate(rating_prediction, topk_prediction, matrix_valid, [metric], [topK], is_topK=is_topK)
            # 如果当前模型的评估指标值大于之前模型的评估指标值，则更新最优模型的参数
            if valid_result[metric][0] > best_result:
                best_result = valid_result[metric][0]
                best_RQ, best_Y = RQ, Y
                result_early_stop = 0
            else:
            # 如果当前的结果没有更好，连续计数器+1，如果连续计数器超过5次，则停止训练
                if result_early_stop > 5:
                    break
                result_early_stop += 1
            #调用 predict() 函数来生成测试集上的评分和topk推荐
            rating_prediction, topk_prediction = predict(matrix_U=RQ,
                                                         matrix_V=Y,
                                                         topK=topK,
                                                         matrix_Train=_matrix_train,
                                                         matrix_Test=matrix_test,
                                                         is_topK=is_topK,
                                                         gpu=gpu_on)
            #计算测试集上的评估指标
            test_result = evaluate(rating_prediction, topk_prediction, matrix_test, [metric], [topK], is_topK=is_topK)
            # 打印训练过程中的评估指标
            for _metric in valid_result.keys():
                # 打印验证集上的评估指标
                print("Epoch {0} Valid {1}:{2}".format(i, _metric, valid_result[_metric]))

            for _metric in test_result.keys():
                # 打印测试集上的评估指标
                print("Epoch {0} Test {1}:{2}".format(i, _metric, test_result[_metric]))
            # 打印损失函数值
            print("Epoch {0} a {1} b {2} c {3}".format(i, a, b, c))

        return best_result, best_RQ, best_X, best_xBias, best_Y.T, best_yBias


def dibmf(matrix_train, matrix_valid, matrix_test, embeded_matrix=np.empty(0), iteration=500, lam=0.01, alpha=0.10273626485512183,
          rank=200, gamma=0.02645887072018175, batch_size=500, learning_rate=0.005, optimizer="Adam", seed=0, gpu_on=False,
          metric='AUC', topK=50, is_topK=False, searcher='optuna', n_trials=100, **unused):
    # 使用 WorkSplitter 模块打印进度信息
    progress = WorkSplitter()
    # 打印“设置随机种子”的信息
    progress.section("DIB-MF: Set the random seed")
    np.random.seed(seed)
    tf.random.set_seed(seed)#tf.set_random_seed(seed)

    progress.section("DIB-MF: Training")
    # 构造一个形状和训练矩阵相同的稀疏矩阵
    temp_matrix_train = csr_matrix(matrix_train.shape)#matrix_train.shape
    # 将训练矩阵中的非零元素置为1
    temp_matrix_train[(matrix_train > 0).nonzero()] = 1
    _matrix_train = temp_matrix_train

    matrix_input = _matrix_train
    if embeded_matrix.shape[0] > 0:
        # 如果嵌入矩阵不为空，将其转置并将其连接到输入矩阵的下方，得到一个新的输入矩阵
        matrix_input = vstack((matrix_input, embeded_matrix.T))
    # 计算新的输入矩阵的形状
    m, n = matrix_input.shape

    if searcher == 'optuna':
        tuner = Tuner()
        trials, best_params = tuner.tune(n_trials=n_trials, num_users=m, num_items=n, optimizer=Regularizer[optimizer],
                                         gpu_on=gpu_on, train=matrix_input, _train=matrix_train, valid=matrix_valid,
                                         test=matrix_test, epoch=iteration, metric=metric, topK=topK, is_topK=is_topK,
                                         seed=seed)
        return trials, best_params

    if searcher == 'grid':
        # 如果使用grid作为超参数搜索器，则创建DIBMF模型对象并进行训练
        model = DIBMF(m, n, rank, batch_size, lamb=lam, alpha=alpha, gamma=gamma, learning_rate=learning_rate,
                      optimizer=Regularizer[optimizer], gpu_on=gpu_on)

        _, RQ, X, xBias, Y, yBias = model.train_model(matrix_input, matrix_train, matrix_valid, matrix_test, iteration,
                                                      metric, topK, is_topK, gpu_on, seed)

        model.sess.close()
        tf.reset_default_graph()

        return RQ, X, xBias, Y, yBias

      
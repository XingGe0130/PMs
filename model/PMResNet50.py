import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers


def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0,
                             min_learn_rate=0,
                             ):
    """
    参数：
        global_step: 上面定义的Tcur，记录当前执行的步数。
        learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
        total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
        warmup_learning_rate: 这是warm up阶段线性增长的初始值
        warmup_steps: warm_up总的需要持续的步数
        hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
    """
    if total_steps < warmup_steps:
        raise ValueError("total_steps must be larger or equal to "
                         "warmup_steps.")
    # 这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
                                                           (global_step - warmup_steps - hold_base_rate_steps) / float(
        total_steps - warmup_steps - hold_base_rate_steps)))
    # 如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError("learning_rate_base must be larger or equal to "
                             "warmup_learning_rate.")
        # 线性增长的实现
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        # 只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)

    learning_rate = max(learning_rate, min_learn_rate)
    return learning_rate


class WarmUpCosineDecayScheduler(callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 min_learn_rate=0,
                 interval_epoch=None,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        # 基础的学习率
        if interval_epoch is None:
            # interval_epoch代表余弦退火之间的最低点
            interval_epoch = [0.05, 0.15, 0.30, 0.50]
        self.learning_rate_base = learning_rate_base
        # 热调整参数
        self.warmup_learning_rate = warmup_learning_rate
        # 参数显示
        self.verbose = verbose
        # learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.min_learn_rate = min_learn_rate
        self.learning_rates = list()

        self.interval_epoch = interval_epoch
        # 贯穿全局的步长
        self.global_step_for_interval = global_step_init
        # 用于上升的总步长
        self.warmup_steps_for_interval = warmup_steps
        # 保持最高峰的总步长
        self.hold_steps_for_interval = hold_base_rate_steps
        # 整个训练的总步长
        self.total_steps_for_interval = total_steps

        self.interval_index = 0
        # 计算出来两个最低点的间隔
        self.interval_reset = [self.interval_epoch[0]]
        for i in range(len(self.interval_epoch) - 1):
            self.interval_reset.append(self.interval_epoch[i + 1] - self.interval_epoch[i])
        self.interval_reset.append(1 - self.interval_epoch[-1])

    # 更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        self.global_step_for_interval = self.global_step_for_interval + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    # 更新学习率
    def on_batch_begin(self, batch, logs=None):
        # 每到一次最低点就重新更新参数
        if self.global_step_for_interval in [0] + [int(i * self.total_steps_for_interval) for i in self.interval_epoch]:
            self.total_steps = self.total_steps_for_interval * self.interval_reset[self.interval_index]
            self.warmup_steps = self.warmup_steps_for_interval * self.interval_reset[self.interval_index]
            self.hold_base_rate_steps = self.hold_steps_for_interval * self.interval_reset[self.interval_index]
            self.global_step = 0
            self.interval_index += 1

        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps,
                                      min_learn_rate=self.min_learn_rate)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print("\nBatch %05d: setting learning "
                  "rate to %s." % (self.global_step + 1, lr))


def identity_block(input_tensor, kernel_size, filters, stage, block, weight_decay=0.):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = layers.Conv1D(filters1, (1,),
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2a")(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters2, kernel_size,
                      padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2b")(x)
    x = layers.BatchNormalization(name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters3, (1,),
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2c")(x)
    x = layers.BatchNormalization(name=bn_name_base + "2c")(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation("relu")(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               weight_decay=0.,
               strides=(2,)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
    # Returns
        Output tensor for the block.
    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = layers.Conv1D(filters1, (1,), strides=strides,
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2a")(input_tensor)
    x = layers.BatchNormalization(name=bn_name_base + "2a")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters2, kernel_size, padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2b")(x)
    x = layers.BatchNormalization(name=bn_name_base + "2b")(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv1D(filters3, (1,),
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(weight_decay),
                      name=conv_name_base + "2c")(x)
    x = layers.BatchNormalization(name=bn_name_base + "2c")(x)

    shortcut = layers.Conv1D(filters3, (1,), strides=strides,
                             kernel_initializer="he_normal",
                             kernel_regularizer=regularizers.l2(weight_decay),
                             name=conv_name_base + "1")(input_tensor)
    shortcut = layers.BatchNormalization(name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def ResNet50(input_shape=None):
    """Instantiates the ResNet50 architecture.
    # Arguments
        input_shape: optional shape tuple, only to be specified
    # Returns
        A Keras model instance.
    """

    input_tensor = Input(shape=input_shape, name="inputs")

    x = layers.Conv1D(256, (3,),
                      strides=(1,),
                      padding="same",
                      kernel_initializer="he_normal",
                      kernel_regularizer=regularizers.l2(0.),
                      name="conv1")(input_tensor)
    x = layers.BatchNormalization(name="bn_conv1")(x)
    x = layers.Activation("relu")(x)
    x = conv_block(x, 3, [256, 256, 128], weight_decay=0., stage=2, block="a", strides=(1,))
    x = identity_block(x, 3, [256, 256, 128], weight_decay=0., stage=2, block="b")
    x = identity_block(x, 3, [256, 256, 128], weight_decay=0., stage=2, block="c")
    x = conv_block(x, 3, [128, 128, 64], weight_decay=0., stage=3, block="a")
    x = identity_block(x, 3, [128, 128, 64], weight_decay=0., stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 64], weight_decay=0., stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 64], weight_decay=0., stage=3, block="d")
    x = conv_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="a")
    x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="b")
    x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="c")
    x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="d")
    x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="e")
    x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=4, block="f")
    # x = conv_block(x, 3, [64, 64, 32], weight_decay=0., stage=5, block="a")
    # x = identity_block(x, 3, [64, 64, 32], weight_decay=0., stage=5, block="b")
    # x = identity_block(x, 3, [256, 256, 512], weight_decay=0., stage=5, block="c")
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(1, name="outputs")(x)

    model = Model(inputs=input_tensor, outputs=x, name="PMResNet50")

    print(model.summary())

    return model


if __name__ == '__main__':
    ResNet50((11, 1))

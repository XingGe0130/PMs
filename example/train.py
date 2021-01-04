import os
import glob
import shutil
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers

from dataset.data_prepare import DataPrepare
from dataset.data_generator import DataGenerator
from dataset.data_prepare import split_files_by_mon
from model.PMResNet50 import ResNet50, WarmUpCosineDecayScheduler

# The GPU id to use, usually either "0" or "1";
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(config_dict):
    data_fp = config_dict.get("data_fp")
    out_fp = config_dict.get("out_fp")
    train_pkl = config_dict.get("train_pkl")
    valid_pkl = config_dict.get("valid_pkl")

    # 参数
    batch_size = config_dict.get("batch_size")
    end_epoch = config_dict.get("end_epoch")
    x_vars = config_dict.get("x_vars")
    y_vars = config_dict.get("y_vars")

    h5_lst = glob.glob(os.path.join(data_fp, "*.HDF"))
    train_files, valid_files, test_files = split_files_by_mon(h5_lst)

    if not os.path.exists(train_pkl):
        train_dp = DataPrepare(train_files, batch_size, "train_flag", x_vars, y_vars)
        train_dp.copy_file(data_fp, os.path.join(out_fp, "hdf", "train"))
        train_dp.write_idx_to_csv(os.path.join(os.path.dirname(train_pkl), "train_idx.csv"))
        train_dp.write_data_to_pkl(data_fp, train_pkl)

    if not os.path.exists(valid_pkl):
        valid_dp = DataPrepare(valid_files, batch_size, "train_flag", x_vars, y_vars)
        valid_dp.copy_file(data_fp, os.path.join(out_fp, "hdf", "valid"))
        valid_dp.write_idx_to_csv(os.path.join(os.path.dirname(valid_pkl), "valid_idx.csv"))
        valid_dp.write_data_to_pkl(data_fp, valid_pkl)

    fd = os.path.join(out_fp, "hdf", "test")
    os.makedirs(fd, exist_ok=True)
    if not os.listdir(fd):
        for f in test_files:
            shutil.copyfile(f, os.path.join(fd, os.path.basename(f)))

    np.random.seed(42)

    # 数据批
    train_batches = DataGenerator(train_pkl, batch_size, shuffle=True)

    valid_batches = DataGenerator(valid_pkl, batch_size, shuffle=False)

    # 模型准备
    model = ResNet50(input_shape=(len(x_vars), 1))

    # model = load_model(r"/path/to/pretrained_weights")

    model.compile(
        optimizer=optimizers.Adam(lr=0.001, decay=1e-4),
        # optimizer=optimizers.SGD(learning_rate=0.001),
        loss=losses.mean_squared_error,
        metrics=[metrics.mean_absolute_error]
    )

    # 训练模型
    log_dir = config_dict.get("log_dir")
    os.makedirs(log_dir, exist_ok=True)

    best_ckpt = callbacks.ModelCheckpoint(
        os.path.join(log_dir, 'weights_best.h5'),
        save_best_only=True,
        monitor='val_mean_absolute_error',
        mode='auto'
    )

    ckpt = callbacks.ModelCheckpoint(
        os.path.join(log_dir, 'weights_{epoch:02d}_{val_mean_absolute_error:.2f}.h5'),
        save_best_only=True,
        monitor='val_mean_absolute_error',
        mode='auto',
        period=10
    )

    train_logger = callbacks.CSVLogger(os.path.join(log_dir, "train_log.csv"))

    # callback = callbacks.LearningRateScheduler(scheduler)

    # early_stopping = callbacks.EarlyStopping(
    #     monitor='val_mean_absolute_error',
    #     patience=10,
    # )

    lr_decay = callbacks.ReduceLROnPlateau(monitor="val_mean_absolute_error", mode="min")

    # warm_up_lr = WarmUpCosineDecayScheduler(
    #     learning_rate_base=0.001,
    #     total_steps=int(end_epoch * len(train_batches) / batch_size),
    #     warmup_learning_rate=1e-5,
    #     warmup_steps=int(5 * len(train_batches) / batch_size),
    #     hold_base_rate_steps=10,
    #     min_learn_rate=1e-6
    # )

    events_dir = os.path.join(log_dir, "events")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    tensorboard = callbacks.TensorBoard(
        log_dir=events_dir,
        histogram_freq=0,
        write_graph=True,
        write_images=True
    )

    model.fit(
        x=train_batches,
        validation_data=valid_batches,
        epochs=end_epoch,
        callbacks=[best_ckpt, ckpt, train_logger, tensorboard]
    )

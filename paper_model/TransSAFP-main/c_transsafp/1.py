import tensorflow as tf

# 导出日志
log_dir = "tmp/logs"
tf.saved_model.load("c_transsafp/models_transfer/amp_transfer_learner.h5")
tf.summary.trace_on(graph=True, profiler=True)
writer = tf.summary.create_file_writer(log_dir)
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0)
writer.close()

# 启动 TensorBoard
# !tensorboard --logdir /tmp/logs

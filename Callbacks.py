import tensorflow as tf

def coef_det_k(y_true, y_pred):
    """Computer coefficient of determination R^2
    """
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + 1e-6)
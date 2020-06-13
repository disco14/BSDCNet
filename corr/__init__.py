#--Ayan Chakrabarti <ayan@wustl.edu>
import tensorflow as tf
import os


sopath=os.path.abspath(os.path.dirname(__file__))+'/corr.so'
mod = tf.load_op_library(sopath)

correlation1d = mod.correlation1d
#_correlation_grad = mod._correlation_grad

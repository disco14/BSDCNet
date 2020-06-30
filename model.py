import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_nn_ops
import corr1d

nfCV = 96 * 3

bneps = 1e-8
ifbn = 1    

encs = [nfCV, 144, 96]
disp = [3, 32, 48, 64, 80, 96]


# Remove BNs (use after replacing filters with popstats)
def toTest():
    global ifbn
    ifbn = 0

def conv(net, name, inp, ksz, bn=0, st=1, relu=True, rt=1):  
    if len(ksz) > 2:
        inch = ksz[2]  
    else:
        inch = int(inp.get_shape()[-1])  

    ksz = [ksz[0], ksz[0], inch, ksz[1]]  

    wnm = name + "_w"  
    if wnm in net.weights.keys():
        w = net.weights[wnm]
    else:
        sq = np.sqrt(3.0 / np.float32(ksz[0] * ksz[1] * ksz[2]))
        w = tf.Variable(tf.random_uniform(ksz, minval=-sq, maxval=sq, dtype=tf.float32))
        net.weights[wnm] = w
        net.wd = net.wd + tf.nn.l2_loss(w)

    if rt != 1:
        out = tf.nn.convolution(inp, w, 'SAME', [st, st], rt) 
    else:
        out = tf.nn.conv2d(inp, w, [1, st, st, 1], 'SAME')

    if bn == 1:
        mu, vr = tf.nn.moments(out, [0, 1, 2]) 
        net.bnvals[name + '_m'] = mu
        net.bnvals[name + '_v'] = vr
        out = tf.nn.batch_normalization(out, mu, vr, None, None, bneps)

    bnm = name + "_b"
    if bnm in net.weights.keys():
        b = net.weights[bnm]
    else:
        b = tf.Variable(tf.constant(0, shape=[ksz[-1]], dtype=tf.float32))
        net.weights[bnm] = b
    out = out + 2.0 * b

    if relu:
        out = tf.nn.relu(out)

    return out


def resconv(net, pfx, inp, out_channel, in_channel, bn, rt=1):
    out = inp

    out1 = conv(net, pfx + '_1', out, [3, out_channel, in_channel], bn, 1, True, rt)
    out1 = conv(net, pfx + '_2', out1, [3, out_channel, out_channel], bn, 1, False, rt)
    out = out + out1
    out = tf.nn.relu(out)

    return out


def downconv(net, pfx, inp, out_channel, in_channel, bn):
    out = inp

    out1 = conv(net, pfx + '_l', out, [1, out_channel, in_channel], bn, 2, False)

    out2 = conv(net, pfx + '_r1', out, [3, out_channel, in_channel], bn, 2, True)
    out2 = conv(net, pfx + '_r2', out2, [3, out_channel, out_channel], bn, 1, False)

    out = out1 + out2
    out = tf.nn.relu(out)

    return out


def fpn(net, pfx, left, right, h_channel, l_channel, bn):
    shap = tf.shape(left)
    left = conv(net, pfx + '_1', left, [1, l_channel, h_channel], bn, 1, True)
    right = conv(net, pfx + '_2', right, [1, l_channel, l_channel], bn, 1, True)

    right = tf.image.resize_bilinear(right, tf.stack([shap[1], shap[2]]))

    out = left + right
    out = conv(net, pfx + '_3', out, [3, l_channel, l_channel], bn, 1, True)

    return out


def ffm(net, pfx, left, right, h_channel, l_channel, bn):
    shap = tf.shape(left)
    right = tf.image.resize_bilinear(right, tf.stack([shap[1], shap[2]]))
    out = tf.concat([left, right], axis=3)
    out = conv(net, pfx + '_1', out, [3, h_channel, h_channel + l_channel], 0, 1, True)

    out1 = conv(net, pfx + '_2', out, [1, h_channel, h_channel], bn, 1, True)

    out1 = tf.reduce_mean(out1, [1, 2], True)
    out1 = conv(net, pfx + '_3', out1, [1, h_channel, h_channel], 0, 1, True) 
    out1 = conv(net, pfx + '_4', out1, [1, h_channel, h_channel], 0, 1, False)
    out1 = tf.nn.sigmoid(out1)

    out2 = out * out1
    out = out + out2

    return out


class Net:
    def __init__(self):
        self.weights = {}
        self.bnvals = {}
        self.wd = 0.

    # Encode images to feature tensor
    def predict(self, img, cv, lrl, rmg):
        fts = cv  
        for i in range(len(encs) - 1):  
            fts = conv(self, 'enc%d' % i, fts, [1, encs[i + 1], encs[i]], ifbn, 1, True)
        fts = resconv(self, 'res2', fts, encs[-1], encs[-1], 0)
        fts = resconv(self, 'res2_2', fts, encs[-1], encs[-1], 0)
      
        disl = conv(self, 'con1_l', img, [7, disp[1], disp[0]], ifbn, 2)  
        disr = conv(self, 'con1_r', rmg, [7, disp[1], disp[0]], ifbn, 2)
        disl = conv(self, 'con2_l', disl, [5, disp[2], disp[1]], ifbn, 2)  
        disr = conv(self, 'con2_r', disr, [5, disp[2], disp[1]], ifbn, 2)

        corr = corr1d.correlation1d(disl, disr, 1, 48, 1, 1,
                                    48)

        corr = resconv(self, 'res6', corr, disp[2], disp[2], 0)
        corr = resconv(self, 'res6_2', corr, disp[2], disp[2], 0)  

        corr2 = downconv(self, 'res7', corr, disp[3], disp[2], 0)
        corr2 = resconv(self, 'res7_2', corr2, disp[3], disp[3], 0)  

        corr3 = downconv(self, 'res8', corr2, disp[4], disp[3], 0)
        corr3 = resconv(self, 'res8_2', corr3, disp[4], disp[4], 0)  

        corr4 = downconv(self, 'res9', corr3, disp[5], disp[4], 0)
        corr4 = resconv(self, 'res9_2', corr4, disp[5], disp[5], 0)  

        corr3_4 = fpn(self, 'fp1', corr3, corr4, disp[4], disp[5], 0)
        corr2_3 = fpn(self, 'fp2', corr2, corr3_4, disp[3], disp[5], 0)
        corr1_2 = fpn(self, 'fp3', corr, corr2_3, disp[2], disp[5], 0)
        corr0_1 = ffm(self, 'ffm', fts, corr1_2, disp[5], disp[5], ifbn)

        out = corr0_1

        out = tf.nn.relu(conv(self, 'out', out, [1, 1, 96], 0, 1, False) + 128.0)

        shp = tf.shape(img)

        out = tf.image.resize_bilinear(out, tf.stack([shp[1], shp[2]]))

        return tf.squeeze(out, axis=-1)




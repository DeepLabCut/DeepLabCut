'''
Adopted: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow

'''

import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses

net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
             'resnet_101': resnet_v1.resnet_v1_101,
             'resnet_152': resnet_v1.resnet_v1_152}

def prediction_layer(cfg, input, name, num_outputs):
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
                        activation_fn=None, normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
        with tf.variable_scope(name):
            pred = slim.conv2d_transpose(input, num_outputs,
                                         kernel_size=[3, 3], stride=cfg.deconvolutionstride,
                                         scope='block4')
            return pred

class PoseNet:
    def __init__(self, cfg):
        self.cfg = cfg
        if 'output_stride' not in self.cfg.keys():
            self.cfg.output_stride=16
        if 'deconvolutionstride' not in self.cfg.keys():
            self.cfg.deconvolutionstride=2

    def extract_features(self, inputs):
        net_fun = net_funcs[self.cfg.net_type]
        mean = tf.constant(self.cfg.mean_pixel,
                           dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        # The next part of the code depends upon which tensorflow version you have.
        vers = tf.__version__
        vers = vers.split(".") #Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44
        if int(vers[0])==1 and int(vers[1])<4: #check if lower than version 1.4.
            with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=self.cfg.output_stride)
        else:
            with slim.arg_scope(resnet_v1.resnet_arg_scope()):
                net, end_points = net_fun(im_centered,
                                          global_pool=False, output_stride=self.cfg.output_stride,is_training=False)

        return net,end_points

    def prediction_layers(self, features, end_points, reuse=None):
        cfg = self.cfg
        num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
        layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

        out = {}
        with tf.variable_scope('pose', reuse=reuse):
            out['part_pred'] = prediction_layer(cfg, features, 'part_pred',
                                                cfg.num_joints)
            if cfg.location_refinement:
                out['locref'] = prediction_layer(cfg, features, 'locref_pred',
                                                 cfg.num_joints * 2)
            if cfg.intermediate_supervision:
                if cfg.net_type=='resnet_50' and cfg.intermediate_supervision_layer>6:
                    print("Changing layer to 6! (higher ones don't exist in block 3 of ResNet 50).")
                    cfg.intermediate_supervision_layer=6
                interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
                block_interm_out = end_points[interm_name]
                out['part_pred_interm'] = prediction_layer(cfg, block_interm_out,
                                                           'intermediate_supervision',
                                                           cfg.num_joints)

        return out

    def get_net(self, inputs):
        net, end_points = self.extract_features(inputs)
        return self.prediction_layers(net, end_points)

    def test(self, inputs):
        heads = self.get_net(inputs)
        prob = tf.sigmoid(heads['part_pred'])
        return {'part_prob': prob, 'locref': heads['locref']}

    def inference(self,inputs):
        ''' Direct TF inference on GPU. Added with: https://arxiv.org/abs/1909.11229'''
        heads = self.get_net(inputs)
        #if cfg.location_refinement:
        locref=heads['locref']
        probs = tf.sigmoid(heads['part_pred'])

        if self.cfg.batch_size==1:
            #assuming batchsize 1 here!
            probs = tf.squeeze(probs, axis=0)
            locref = tf.squeeze(locref, axis=0)
            l_shape = tf.shape(probs)

            locref = tf.reshape(locref, (l_shape[0]*l_shape[1], -1, 2))
            probs = tf.reshape(probs , (l_shape[0]*l_shape[1], -1))
            maxloc = tf.argmax(probs, axis=0)

            loc = tf.unravel_index(maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64)))
            maxloc = tf.reshape(maxloc, (1, -1))

            joints = tf.reshape(tf.range(0, tf.cast(l_shape[2], dtype=tf.int64)), (1,-1))
            indices = tf.transpose(tf.concat([maxloc,joints] , axis=0))

            offset = tf.gather_nd(locref, indices)
            offset = tf.gather(offset, [1,0], axis=1)
            likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1,1))

            pose = self.cfg.stride*tf.cast(tf.transpose(loc), dtype=tf.float32) + self.cfg.stride*0.5 + offset*self.cfg.locref_stdev
            pose = tf.concat([pose, likelihood], axis=1)

            return {'pose': pose}
        else:
            #probs = tf.squeeze(probs, axis=0)
            l_shape = tf.shape(probs) #batchsize times x times y times body parts
            #locref = locref*cfg.locref_stdev
            locref = tf.reshape(locref, (l_shape[0],l_shape[1],l_shape[2],l_shape[3], 2))
            #turn into x times y time bs * bpts
            locref=tf.transpose(locref,[1,2,0,3,4])
            probs=tf.transpose(probs,[1,2,0,3])

            #print(locref.get_shape().as_list())
            #print(probs.get_shape().as_list())
            l_shape = tf.shape(probs) # x times y times batch times body parts

            locref = tf.reshape(locref, (l_shape[0]*l_shape[1], -1, 2))
            probs = tf.reshape(probs , (l_shape[0]*l_shape[1],-1))
            maxloc = tf.argmax(probs, axis=0)
            loc = tf.unravel_index(maxloc, (tf.cast(l_shape[0], tf.int64), tf.cast(l_shape[1], tf.int64))) #tuple of max indices

            maxloc = tf.reshape(maxloc, (1, -1))
            joints = tf.reshape(tf.range(0, tf.cast(l_shape[2]*l_shape[3], dtype=tf.int64)), (1,-1))
            indices = tf.transpose(tf.concat([maxloc,joints] , axis=0))

            #extract corresponding locref x and y as well as probability
            offset = tf.gather_nd(locref, indices)
            offset = tf.gather(offset, [1,0], axis=1)
            likelihood = tf.reshape(tf.gather_nd(probs, indices), (-1,1))

            pose = self.cfg.stride*tf.cast(tf.transpose(loc), dtype=tf.float32) + self.cfg.stride*0.5 + offset*self.cfg.locref_stdev
            pose = tf.concat([pose, likelihood], axis=1)
            return {'pose': pose}

    def train(self, batch):
        cfg = self.cfg

        heads = self.get_net(batch[Batch.inputs])

        weigh_part_predictions = cfg.weigh_part_predictions
        part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0

        def add_part_loss(pred_layer):
            return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
                                                   heads[pred_layer],
                                                   part_score_weights)

        loss = {}
        loss['part_loss'] = add_part_loss('part_pred')
        total_loss = loss['part_loss']
        if cfg.intermediate_supervision:
            loss['part_loss_interm'] = add_part_loss('part_pred_interm')
            total_loss = total_loss + loss['part_loss_interm']

        if cfg.location_refinement:
            locref_pred = heads['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
            loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
            total_loss = total_loss + loss['locref_loss']

        # loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
        loss['total_loss'] = total_loss
        return loss

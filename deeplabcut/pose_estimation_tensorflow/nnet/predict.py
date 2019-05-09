'''
Adapted from original predict.py by Eldar Insafutdinov's implementation of [DeeperCut](https://github.com/eldar/pose-tensorflow)
To do faster inference on videos. See https://www.biorxiv.org/content/early/2018/10/30/457242

Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import numpy as np
import tensorflow as tf
from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net

# Added methods for supporting saving the probability frames:

from io import StringIO
from typing import TextIO

# Class is used so we can have a counter variable which is static and is not reset every execution
class FrameWriter:
    current_frame = 1
    
    @staticmethod
    def write_frame(scmap, cfg, save_file: str) -> None:
        """
        Writes the given source map of probabilities to the specified file
        
        Format written to file is the frame, followed by the bodypart, followed by a list of probabilities, ending in a ////.
        
        param scmap: The source map of the given image or frame of video, the probability frame.
        param cfg: The dlc_cfg, used to get the names of the bodyparts
        param save_file: The file location to save to, should end with .csv(sorta makes a csv)
                         NOTE: This is a string path, not a file handler...
        """
        # If there is more then 4 dimensions, there is a problem
        if(len(scmap.shape) > 4):
            raise ValueError("scmap has more then 4 dimensions!")
        # If we have 4 dimensions in the source map, we have multiple frames of the video, 
        # so recall this function with each frame seperated
        if(len(scmap.shape) > 3):
            for innermap in scmap:
                FrameWriter.write_frame(innermap, cfg, save_file)
            return
        # Iterating the last row in each 3-dimensional array list
        # As this is a specific bodypart of the degu
        for i in range(scmap.shape[-1]):
            # Open file path as f, and append to the end...
            with open(save_file, "a") as f:
                # Create StringIO to store output of numpy savetxt
                temp = StringIO()
                # Write the current frame and bodypart
                temp.write(f"Frame: {FrameWriter.current_frame}\n")
                temp.write(f"BodyPart: {cfg['all_joints_names'][i]}, {i}\n")
                # Write probability data for this bodypart, we also have to reshape data 
                # to appear like the original frame dimensions... 
                np.savetxt(temp, scmap[:, :, i].reshape(scmap.shape[-3], scmap.shape[-2]))
                # Write //// to indicate end of this block of data
                temp.write("////\n")
                
                # Write the string buffer
                f.write(temp.getvalue())
                # Close temp string buffer
                temp.close()
        # Increment the frame...
        FrameWriter.current_frame += 1


# ORIGINAL METHODS BELOW:

def setup_pose_prediction(cfg):
    tf.reset_default_graph()
    inputs = tf.placeholder(tf.float32, shape=[cfg.batch_size   , None, None, 3])
    net_heads = pose_net(cfg).test(inputs)
    outputs = [net_heads['part_prob']]
    if cfg.location_refinement:
        outputs.append(net_heads['locref'])

    restorer = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # Restore variables from disk.
    restorer.restore(sess, cfg.init_weights)

    return sess, inputs, outputs
    
def extract_cnn_output(outputs_np, cfg):
    ''' extract locref + scmap from network '''
    scmap = outputs_np[0]
    scmap = np.squeeze(scmap)
    locref = None
    if cfg.location_refinement:
        locref = np.squeeze(outputs_np[1])
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1], -1, 2))
        locref *= cfg.locref_stdev
    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref

def argmax_pose_predict(scmap, offmat, stride):
    """Combine scoremat and offsets to the final pose."""
    num_joints = scmap.shape[2]
    pose = []
    for joint_idx in range(num_joints):
        maxloc = np.unravel_index(np.argmax(scmap[:, :, joint_idx]),
                                  scmap[:, :, joint_idx].shape)
        offset = np.array(offmat[maxloc][joint_idx])[::-1]
        pos_f8 = (np.array(maxloc).astype('float') * stride + 0.5 * stride +
                  offset)
        pose.append(np.hstack((pos_f8[::-1],
                               [scmap[maxloc][joint_idx]])))
    return np.array(pose)

# Arg h5_path added
def getpose(image, cfg, sess, inputs, outputs, h5_path, outall=False):
    ''' Extract pose '''
    im=np.expand_dims(image, axis=0).astype(float)
    outputs_np = sess.run(outputs, feed_dict={inputs: im})
    scmap, locref = extract_cnn_output(outputs_np, cfg)
    
    # This line is added to save the scmaps
    FrameWriter.write_frame(scmap, cfg, h5_path[:-3])
    
    pose = argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose

## Functions below implement are for batch sizes > 1:
def extract_cnn_outputmulti(outputs_np, cfg):
    ''' extract locref + scmap from network 
    Dimensions: image batch x imagedim1 x imagedim2 x bodypart'''
    scmap = outputs_np[0]
    locref = None
    if cfg.location_refinement:
        locref =outputs_np[1]
        shape = locref.shape
        locref = np.reshape(locref, (shape[0], shape[1],shape[2], -1, 2))
        locref *= cfg.locref_stdev
    if len(scmap.shape)==2: #for single body part!
        scmap=np.expand_dims(scmap,axis=2)
    return scmap, locref

# Arg video_path added
def getposeNP(image, cfg, sess, inputs, outputs, h5_path, outall=False):
    ''' Adapted from DeeperCut, performs numpy-based faster inference on batches'''
    outputs_np = sess.run(outputs, feed_dict={inputs: image})
    
    scmap, locref = extract_cnn_outputmulti(outputs_np, cfg) #processes image batch.
    
    # Line added to save srcmaps
    FrameWriter.write_frame(scmap, cfg, h5_path[:-3] + ".csv")
    
    batchsize,ny,nx,num_joints = scmap.shape
    
    #Combine scoremat and offsets to the final pose.
    LOCREF=locref.reshape(batchsize,nx*ny,num_joints,2)
    MAXLOC=np.argmax(scmap.reshape(batchsize,nx*ny,num_joints),axis=1)
    Y,X=np.unravel_index(MAXLOC,dims=(ny,nx))
    DZ=np.zeros((batchsize,num_joints,3))
    for l in range(batchsize):
        for k in range(num_joints):
            DZ[l,k,:2]=LOCREF[l,MAXLOC[l,k],k,:]
            DZ[l,k,2]=scmap[l,Y[l,k],X[l,k],k]
            
    X=X.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,0]
    Y=Y.astype('float32')*cfg.stride+.5*cfg.stride+DZ[:,:,1]
    pose = np.empty((cfg['batch_size'], cfg['num_joints']*3), dtype=X.dtype) 
    pose[:,0::3] = X
    pose[:,1::3] = Y
    pose[:,2::3] = DZ[:,:,2] #P
    if outall:
        return scmap, locref, pose
    else:
        return pose


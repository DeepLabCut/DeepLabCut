"""
DeepLabCut2.2 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import numpy as np

###################################
#### auxiliaryfunctions
###################################

def distance(v,w):
    return np.sqrt(np.sum((v-w)**2))

def minmax(array,slack=10):
    return np.nanmin(array)-slack,np.nanmax(array)+slack

def individual2boundingbox(cfg,animals,X1=0):
    boundingboxes=np.zeros((len(animals),5))*np.nan

    for id,individual in enumerate(animals):
        boundingboxes[id,0:4:2]=minmax(individual[::3]+X1,slack=cfg.boundingboxslack)
        boundingboxes[id,1:4:2]=minmax(individual[1::3],slack=cfg.boundingboxslack)
        boundingboxes[id,4]=np.nanmean(individual[2::3]) #average likelihood of all bpts!

    return boundingboxes

##########################################################
#### conversion & greedy bodypart matching code
##########################################################

def convertdetectiondict2listoflist(dataimage,imname,BPTS,withid=False,evaluation=False):
    ''' Arranges data into list of list with the following entries:
    [(x, y, score, global index of detection)] (all detections per bodypart).

    Also includes id if available. [x,y,score,global id, id] '''
    if evaluation:
        detectedcoordinates=dataimage['prediction']['coordinates'][0]
        detectedlikelihood=dataimage['prediction']['confidence']
    else:
        detectedcoordinates=dataimage['coordinates'][0]
        detectedlikelihood=dataimage['confidence']

    all_detections=[]
    detection_counter=0
    for bpt in BPTS:
        if withid: #(x, y, likelihood, identity)
            detections_with_likelihood = list(zip(detectedcoordinates[bpt][:,0], detectedcoordinates[bpt][:,1],
                                        detectedlikelihood[bpt].flatten(),np.argmax(dataimage['identity'][bpt],1)))
        else: #(x, y, likelihood)
            detections_with_likelihood = list(zip(detectedcoordinates[bpt][:,0], detectedcoordinates[bpt][:,1],detectedlikelihood[bpt].flatten()))
        idx = range(detection_counter, detection_counter+ len(detections_with_likelihood))
        all_detections.append([detections_with_likelihood[i] + (idx[i],) for i in range(len(idx))])
        detection_counter += len(detections_with_likelihood)

    return all_detections

def matchconnectionsevaluation(cfg, dataimage, all_detections, iBPTS, partaffinityfield_graph, PAF):
    ''' Auxiliary function;  Returns list of connections (limbs) of a particular type.

    Specifically, per edge a list containing is returned: [index start (global), index stop (global) score, score with detection likelihoods, index start (local), index stop (local)]
    Thereby, index start and stop refer to the index of the bpt from the beginning to the end of the edge. Local index refers to the index within the list of bodyparts, and
    global to the index for all the detections (in all_detections)

    Parameters
    ----------
    cfg : dictionary
        configuation file for inference parameters

    dataimage: dict
        predictions (detections + paf scores) for a particular image

    all_detections: list of lists
        result of convertdetectiondict2listoflist

    iBPTS: list
        source of bpts.

    partaffinityfield_graph: list of part affinity matchconnections

    PAF: PAF is a subset of the indices that should be used in the partaffinityfield_graph

    '''
    all_connections = []
    missing_connections = []
    for k in range(len(partaffinityfield_graph)):
        cand_a = all_detections[iBPTS[partaffinityfield_graph[k][0]]] # transfer orignal bpt index to the one in all_detections!
        cand_b = all_detections[iBPTS[partaffinityfield_graph[k][1]]] #
        n_a = len(cand_a)
        n_b = len(cand_b)
        if n_a != 0 and n_b != 0:
            connection_candidate = []
            for i in range(n_a):
                for j in range(n_b):
                    KK=PAF[k]
                    d=distance(np.array(cand_a[i][:2]),np.array(cand_b[j][:2]))
                    score_with_dist_prior=abs(dataimage['prediction']['costs'][KK][cfg.method][i,j])
                    si=cand_a[i][2] #detectedlikelihood[partaffinityfield_graph[k][0]][i].flatten()[0]
                    sj=cand_b[j][2] #detectedlikelihood[partaffinityfield_graph[k][1]][j].flatten()[0]
                    if score_with_dist_prior>cfg.pafthreshold and d<cfg.distnormalization and d>=cfg.distnormalizationLOWER and si*sj>cfg.detectionthresholdsquare:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + np.sqrt(si * sj)*cfg.addlikelihoods])

            #sort candidate connections by score!
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    ii=int(cand_a[i][-1]) #global index!
                    jj=int(cand_b[j][-1])
                    connection = np.vstack([connection, [ii, jj, s, i, j]])
                    if len(connection) >= min(n_a, n_b):
                        break

            all_connections.append(connection)
        else:
            missing_connections.append(k)
            all_connections.append([])
    return all_connections, missing_connections

def matchconnections(cfg,dataimage,all_detections,iBPTS,partaffinityfield_graph,PAF):
    ''' PAF is a subset of indices should be used in the PAF graph '''
    all_connections = []
    missing_connections = []
    for k in range(len(partaffinityfield_graph)):
        cand_a = all_detections[iBPTS[partaffinityfield_graph[k][0]]] # detectedcoordinates[partaffinityfield_graph[k][0]]
        cand_b = all_detections[iBPTS[partaffinityfield_graph[k][1]]] # detectedcoordinates[partaffinityfield_graph[k][1]]
        n_a = len(cand_a)
        n_b = len(cand_b)
        if n_a != 0 and n_b != 0:
            connection_candidate = []
            for i in range(n_a):
                for j in range(n_b):
                    KK=PAF[k]
                    d=distance(np.array(cand_a[i][:2]),np.array(cand_b[j][:2]))
                    score_with_dist_prior=abs(dataimage['costs'][KK][cfg.method][i,j])
                    si=cand_a[i][2] #detectedlikelihood[partaffinityfield_graph[k][0]][i].flatten()[0]
                    sj=cand_b[j][2] #detectedlikelihood[partaffinityfield_graph[k][1]][j].flatten()[0]
                    if score_with_dist_prior>cfg.pafthreshold and d<cfg.distnormalization and d>cfg.distnormalizationLOWER and si*sj>cfg.detectionthresholdsquare:
                        connection_candidate.append([i, j, score_with_dist_prior,
                                                     score_with_dist_prior + np.sqrt(si * sj)*cfg.addlikelihoods])

            #sort by score!
            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0, 5))
            for c in range(len(connection_candidate)):
                i, j, s = connection_candidate[c][0:3]
                if i not in connection[:, 3] and j not in connection[:, 4]:
                    ii=int(cand_a[i][-1]) #global index!
                    jj=int(cand_b[j][-1])
                    connection = np.vstack([connection, [ii, jj, s, i, j]])
                    if len(connection) >= min(n_a, n_b):
                        break

            all_connections.append(connection)
        else:
            missing_connections.append(k)
            all_connections.append([])
    return all_connections, missing_connections

def linkjoints2individuals(cfg,all_detections,all_connections, missing_connections,partaffinityfield_graph,iBPTS,numjoints=18):
    subset = np.empty((0, numjoints+2))
    candidate = np.array([item for sublist in all_detections for item in sublist])
    for k in range(len(partaffinityfield_graph)):
        if k not in missing_connections:
            part_as = all_connections[k][:, 0] #global? indices for source of limb!
            part_bs = all_connections[k][:, 1] #indices for target of limb (when all detections are enumerated)
            index_a, index_b =iBPTS[partaffinityfield_graph[k][0]],iBPTS[partaffinityfield_graph[k][1]] #convert in order of bpts!
            for i in range(len(all_connections[k])):  # looping over all connections for that limb
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)):  # current number of individuals...
                    if (subset[j][index_a] == part_as[i] or subset[j][index_b] == part_bs[i]) and found<2: #added found<2
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if subset[j][index_b] != part_bs[i]: #WHAT ABOUT index_a??? >>>MAKE ELSE!!
                        subset[j][index_b] = part_bs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[part_bs[i].astype(int), 2] + all_connections[k][i][2]
                elif found == 2:  # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    membership = ((subset[j1] >= 0).astype(int) + (subset[j2] >= 0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0:  # merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += all_connections[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else:  # as like found == 1 IE >>> LINKS PART B!?!. This sounds like a bad idea for non trees!
                        subset[j1][index_b] = part_bs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[part_bs[i].astype(int), 2] + all_connections[k][i][2]
                        #WHAT ABOUT j2?

                # if find no partA in the subset, create a new subset
                elif not found and k < numjoints:
                    row = -1 * np.ones(numjoints+2)
                    row[index_a] = part_as[i]
                    row[index_b] = part_bs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[all_connections[k][i, :2].astype(int), 2]) + all_connections[k][i][2]
                    subset = np.vstack([subset, row])

    deleteIdx = [];
    for i in range(len(subset)):  # delete animals with too few connected bodyparts or low average score
        if subset[i][-1] < cfg.minimalnumberofconnections or subset[i][-2]/subset[i][-1] < cfg.averagescore:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)
    #subset = sorted(subset, key=lambda x: x[-2], reverse=True)
    return subset, candidate

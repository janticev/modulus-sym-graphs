import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import pdb
import numpy as np
import os 
import time
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
import modulus.external.SPADE_score.SPADE_nxFree as SPADE
from modulus.external.Clustering.C_HyperEF import C_HyperEF
from modulus.external.Clustering.C_sim_coarse import C_sim_coarse
import modulus.external.SGM_SPADE_utils as GraphUtils
import modulus.external.Clustering.gz_utils as GzUtils
import gc
from sklearn import preprocessing
from functools import reduce

def combineGrids_helper(Cn, binned_refs = None, projection = None):
        rows = 0
        cols = 0
        rowShift = []
        colShift = []
        for i in Cn:
            rowShift.append(rows)
            rows += i.shape[0]
            colShift.append(cols)
            cols += i.shape[1]
        print([rows,cols])
        print('rowsS',rowShift,'colsS',colShift)
        rowList = []
        colList = []
        dataList = []
        for k,i in enumerate(Cn):
            hold = i.tocoo(copy=True)
            #print('row',hold.row, len(hold.row))
            if not binned_refs:
                #print('row',hold.row, len(hold.row))
                rowList.append(hold.row+rowShift[k])
                newCol = hold.col+colShift[k]
                #print('col',[min(newCol), max(newCol),len(newCol), colShift[k]])
                colList.append(newCol)
                dataList.append(hold.data)
            else:
                #newRow = binned_refs[k][hold.row]
                #print('row2',[min(newRow), max(newRow),len(newRow), colShift[k]])
                #print('brefs',len(binned_refs),binned_refs[k].shape)
                rowList.append(binned_refs[k][hold.row])
                if projection == 'proj':
                    newCol = hold.col+colShift[k]
                elif projection == 'lap':
                    newCol = binned_refs[k][hold.col]
                else:
                    raise ValueError
                #print('col',[min(newCol), max(newCol),len(newCol), colShift[k]])
                colList.append(newCol)
                dataList.append(hold.data)
        cc = lambda x: np.concatenate(x,axis=0)
        return coo_matrix((cc(dataList), (cc(rowList),cc(colList))),shape=(rows,cols)).tocsc()

def dict_to_numpy(invar):
        data = np.concatenate([invar[x] for x in invar.keys()], axis=1)
        num_elements = list(invar.values())[0].shape[0]
        ids = np.arange(num_elements)
        lkeys = invar.keys()
        dim = len(lkeys)
        return data, ids, lkeys, num_elements, dim

def idxLists_to_matrix_level(idx_mat, L):
    nclusters = np.max(idx_mat[L-1]+1)
    idx = np.arange(np.max(nclusters)) #the number of nodes in the coarsest graph
    for i in reversed(range(L)):
        idx = idx[idx_mat[i]] #list of node's cluster at level L
    row = np.arange(len(idx))
    col = idx
    P = csc_matrix((np.ones(len(idx),dtype='int'),(row,col)), shape = (len(idx), nclusters))
    return P

def SPADE_uniform_quantile(scores):
        return preprocessing.normalize(
                                [preprocessing.quantile_transform(
                                    scores.reshape(-1, 1), 
                                    n_quantiles=300, 
                                    output_distribution='uniform', 
                                    ignore_implicit_zeros=False, 
                                    subsample=100000, 
                                    random_state=None).reshape(-1,)],
                                norm='l2', 
                                axis=1, copy=True, return_norm=False
                            ).reshape(-1,)**2

def send_data_protected(wID,pwID,return_pipe,data):
    print(f"{pwID} Pushing to main Pipe",flush = True)
    if return_pipe.poll(timeout = .1):
        print(f"{pwID} Pipe Gone, exiting",flush = True)
        sys.exit() #Main doesn't send data here, pipe reports a value ready if the other end dies. If value ready on this end, main died.
    #return_pipe.send(np.array([wID,data], dtype="object")) ##NOTE blocking call... Should eventually be replaced
    return_pipe.send(([wID,data])) ##NOTE blocking call... Should eventually be replaced
    print(f"{pwID} Push Success", flush = True)

def worker_HyperEF_fn(wID, detail,                      
                        dataset, 
                        return_pipe, 
                        knn, #sparsification,
                        level, r, 
                        initial_graph_vars,
                        graph_vars
                    ):
        pwID = f"CLUSTER_{wID}"
        JLoad = C_HyperEF()
        start = time.time()
        print(f"{pwID} Opened",flush=True)
        print(f"{pwID} Creating Subgraph",flush=True)
        print(f"{pwID} Dataset get  {type(dataset)} {type(dataset['x'])}",flush=True)
        if detail == 'initial':
            construct_vars = initial_graph_vars
            print(f"{pwID} Detail message get {detail}")
        else:
            construct_vars = graph_vars
        clusters = JLoad.HyperEF_P(dataset,
            construct_vars,
            knn,
            level)
        #print(f'{pwID} CLUSTERS::: {clusters.shape}', flush = True)
        send_data_protected(wID,pwID,return_pipe,clusters)
        end = time.time()
        print(f'{pwID} TIME TAKEN {end-start}', flush = True)

def worker_GraphZoom_fn(wID, detail,                      
                        dataset, 
                        return_pipe, 
                        knn, #sparsification,
                        level, r, 
                        initial_graph_vars,
                        graph_vars
                    ):
        pwID = f"CLUSTER_{wID}"
        JLoad = C_sim_coarse()
        start = time.time()
        print(f"{pwID} Opened",flush=True)
        print(f"{pwID} Creating Subgraph",flush=True)
        print(f"{pwID} Dataset get  {type(dataset)} {type(dataset['x'])}",flush=True)
        if detail == 'initial':
            construct_vars = initial_graph_vars
            print(f"{pwID} Detail message get {detail}")
        else:
            construct_vars = graph_vars
        clusters = JLoad.sim_coarse_call(dataset,
            construct_vars,
            knn,
            level)
        #print(f'{pwID} CLUSTERS::: {clusters.shape}', flush = True)
        send_data_protected(wID,pwID,return_pipe,clusters)
        end = time.time()
        print(f'{pwID} TIME TAKEN {end-start}', flush = True)


def worker_SPADE_fn(wID,
                    detail,
                    dataset_input,
                    dataset_output,
                    return_pipe, 
                    knn, num_eigs,
                    initial_graph_vars,
                    output_graph_vars,
                    ):
        start = time.time()
        pwID = f"SPADE_{wID}"
        print(f"{pwID} Opened",flush=True)
        print(f"{pwID} Creating Subgraph",flush=True)
        #Create new sub graph
        print(f"{pwID} Dataset input get  {type(dataset_input)} {dataset_input[list(dataset_input.keys())[0]]}",flush=True)
        print(f"{pwID} Dataset output get  {type(dataset_output)} {dataset_output[list(dataset_output.keys())[0]]}",flush=True)
        print(f"{pwID} Dataset output keys: {list(dataset_output.keys())}")
        print(f"{pwID} Dataset initial_graph_vars:{initial_graph_vars}")
        print(f"{pwID} Dataset output_graph_vars:{output_graph_vars}")
        if detail == 'initial':
            dataset_input = {k:v.cpu().detach().numpy() for k,v in dataset_input.items() if k in initial_graph_vars}
            print(f"{pwID} Dataset Input Detached, Initial Graph Vars",flush=True)
        else:
            dataset_input = {k:v for k,v in dataset_input.items() if k in output_graph_vars}
            print(f"{pwID} Dataset Input As Received, main Graph Vars",flush=True)
        start2 = time.time()
        dataset_input, _, lkeys, num_elements, dim = dict_to_numpy(dataset_input)
        end2 = time.time()
        print(f'{pwID} DICT_TO_NUMPY_INPUT {end2-start2}', flush = True)
        print(f'{pwID}Keys:' + str(lkeys))
        print(f'Size:({num_elements}x{dim})')
        dataset_output = {k:v for k,v in dataset_output.items() if k in output_graph_vars}
        dataset_output, _, lkeys, num_elements, dim = dict_to_numpy(dataset_output)
        print(f'{pwID}Using: {str(lkeys)}, KNN: {knn}, num_eigs: {num_eigs}')
        print(f'{pwID}Keys:' + str(lkeys))
        print(f'Size:({num_elements}x{dim})')

        dataset_input = preprocessing.scale(dataset_input)
        dataset_output = preprocessing.scale(dataset_output)
        
        spadeOut = SPADE.spade(dataset_input, dataset_output, 
                            k=knn, 
                            num_eigs = num_eigs,
                            sparse = False, weighted=False,
                            wID = wID)[:4]
        
        spadeOut = np.array(spadeOut, dtype="object")
        
        outs = ['TopEig', 'TopEdgeList', 'TopNodeList', 'node_score', 'L_in', 'L_out', 'Dxy', 'Uxy']
        
        send_data_protected(wID,pwID,return_pipe,spadeOut)
        end = time.time()
        print(f'{pwID} TIME TAKEN {end-start}', flush = True)


def worker_BATCH_fn(wID,
                    embeddings,
                    batch_size,
                    return_pipe,
                    recieve_pipe
                    ):
        while True:
            pwID = f"BATCH_{wID}"
            print(f"{pwID} Opened",flush=True)  
            #prob = SPADE_uniform_quantile(embeddings)
            #prob = preprocessing.normalize(embeddings,norm='l2',axis=0)[:,0]**2 ##1D
            #prob_n = preprocessing.normalize(embeddings,norm='l2',axis=0)[:,0] ##1D
            prob = (embeddings/np.sum(embeddings))[:,0]
            print(f"%%%%%% PROB: {prob.shape} %%%%%%")
            idx_size = embeddings.shape[0]//16
            div_size = idx_size//4
            while True:
                start = time.time()
                batches = np.array([])
                for i in range(int(idx_size/div_size)):
                    batches = np.concatenate([batches, np.random.choice(embeddings.shape[0], 
                                            div_size, replace=False, p = prob)]) 
                # batches = np.random.choice(embeddings.shape[0],
                #                        int(embeddings.shape[0]*.05),
                #                        replace=False,
                #                        p=prob)
                batches = batches.astype(np.int64)[:,None]
                message = (batches, prob)
                end = time.time()
                print(f'{pwID} BATCH TIME TAKEN {end-start}', flush = True)
                send_data_protected(wID,pwID,return_pipe, message)
                if recieve_pipe.poll(timeout = .1):
                    break

# def worker_BATCH_fn_OLD(wID,
#                     coarse_LOSS, 
#                     coarse_SPADE,
#                     Ps,
#                     return_pipe,
#                     recieve_pipe
#                     ):
#         while True:
            
#             pwID = f"BATCH_{wID}"
#             print(f"{pwID} Opened",flush=True)
#             print(f"{pwID} Creating Subgraph",flush=True)
#             print(f"{pwID} P get  {type(Ps)} {Ps}",flush=True)
#             print(f"{pwID} LOSS get {type(coarse_LOSS)} {coarse_LOSS.shape}",flush=True)
#             print(f"{pwID} SPADE get {coarse_SPADE}",flush=True)
#             ##TODO, need to propegate coarse scores to all samples using Ps, see refinement step...
#             # ##First try, LOSS SUM,SPADE, sum at end
#             # coarse_SPADE = coarse_SPADE[:,None]
#             # #coarse_SPADE = preprocessing.scale(coarse_SPADE,with_mean=False)
#             # coarse_LOSS = np.abs(coarse_LOSS) #ensure positive
#             # coarse_LOSS = preprocessing.scale(coarse_LOSS,with_mean=False)
#             # coarse_LOSS = np.sum(coarse_LOSS, axis=1)[:,None] #sum losses of each sample
#             # coarse_LOSS = preprocessing.normalize(coarse_LOSS, axis=0)**2
#             # embeddings = np.hstack([coarse_LOSS,coarse_SPADE])
#             # projections = Ps[0]
#             # laplacians = Ps[1]
#             # level = Ps[2]
#             # ######Refinement######
#             # print("%%%%%% Starting Graph Refinement %%%%%%")
#             # refine_start = time.process_time()
#             # embeddings   = GzUtils.refinement(level, projections, laplacians, embeddings) #TODO lda/power??
#             # refine_time  = time.process_time() - refine_start
#             # print(f"%%%%%% Refine time:{refine_time} %%%%%%")
#             # scores = np.sum(embeddings, axis=1)[:,None]
#             # print(f"%%%%%% SCORES: {scores.shape} %%%%%%")
#             # ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#             ##Second try: LOSS+SPADE as 1 score, then scaled (SPADE is equal weight to any loss term)
#             start = time.time()
#             embeddings = coarse_LOSS
#             embeddings = np.abs(embeddings) #ensure positive
#             embeddings = preprocessing.scale(embeddings,with_mean=False)
#             embeddings = np.sum(embeddings, axis=1)[:,None] #sum losses of each sample
#             np.savez('cluster_scores',embeddings)
#             #coarse_LOSS = preprocessing.normalize(coarse_LOSS, axis=0)**2
#             projections = Ps[0]
#             laplacians = Ps[1]
#             level = Ps[2]
#             ######Refinement######
#             print("%%%%%% Starting Graph Refinement %%%%%%")
#             refine_start = time.process_time()
#             embeddings   = GzUtils.refinement(level, projections, laplacians, embeddings) #TODO lda/power??
#             refine_time  = time.process_time() - refine_start
#             print(f"%%%%%% Refine time:{refine_time} %%%%%%")
#             #scores = np.sum(embeddings, axis=1)[:,None]
#             np.savez('sample_scores',embeddings)
#             print(f"%%%%%% SCORES: {embeddings.shape} %%%%%%")
#             prob = SPADE_uniform_quantile(embeddings)
#             #prob = preprocessing.normalize(embeddings,norm='l2',axis=0)[:,0]**2 ##1D
#             print(f"%%%%%% PROB: {prob.shape} %%%%%%")
#             end = time.time()
#             print(f'{pwID} TIME TAKEN {end-start}', flush = True)
#             with open('./logging.txt', 'a+') as f:
#                 f.write(f'New Batch Scores, {end-start} \n')
#             while True:
#                 start = time.time()
#                 batches = np.random.choice(embeddings.shape[0],
#                                         (int(embeddings.shape[0]*.3),1),
#                                         replace=False,
#                                         p=prob)
#                 end = time.time()
#                 print(f'{pwID} BATCH TIME TAKEN {end-start}', flush = True)
#                 send_data_protected(wID,pwID,return_pipe, batches)
#                 if recieve_pipe.poll(timeout = .1):
#                     break

import multiprocessing as mp
class MultiProcessGZ:
    
    def __init__(self, dataset, batch_size, 
                 knn, initial_graph_vars, graph_vars, 
                 SPADE_vars, LOSS_vars,
                 grid_width,
                 num_eigs,
                 SPADE_GRID,
                 level,
                 r,
                 sample_ratio,
                 sample_bounds
                 ):
        self.dataset = dataset
        np.savez_compressed('initialization_dataset',**dataset)
        self.dataset_output = dataset
        self.batch_size = batch_size
        self.knn = knn
        self.num_eigs = num_eigs
        self.SPADE_GRID = SPADE_GRID    

        self.level = level
        self.r = r
        self.sample_ratio=sample_ratio
        self.sample_bounds=sample_bounds

        self.initial_graph_vars = initial_graph_vars
        self.graph_vars = graph_vars
        self.SPADE_vars = SPADE_vars
        print(f'SPADE_vars: {self.SPADE_vars}')
        self.LOSS_vars = LOSS_vars
        print(f'LOSS_vars: {self.LOSS_vars}')
        self.grid_width = grid_width

        
        self.binned_datasets_refs = self.split_grids(self.dataset, self.grid_width) # this would be a list of lists of indexes for samples in each grid space
        
        for k,i in enumerate(self.binned_datasets_refs):
            np.savez(f'{k}' +'_initialRefs.npz', i)

        self.num_workers = len(self.binned_datasets_refs)
        self.SPADE_num_workers = self.num_workers**SPADE_GRID

        self.CLUSTER_workers = [None for _ in range(self.num_workers)] #workers for graph construction
        self.SPADE_workers = [None for _ in range(self.SPADE_num_workers)] #workers for SPADE calcs
        self.BATCH_workers = [None for _ in range(2)] #workers for creating sampling batches

        self.binned_datasets_in = []
        self.binned_datasets_out = []

        self.SPADE_binned_results = [None for _ in range(self.SPADE_num_workers)]
        self.SPADE_combined_results = None

        self.CLUSTER_binned_results = [None for _ in range(self.num_workers)]
        self.CLUSTER_combined_results = None
 
        self.CLUSTER_pipes_recv = [None for _ in range(self.num_workers)]
        self.SPADE_pipes_recv = [None for _ in range(self.SPADE_num_workers)]
        self.BATCH_pipes_recv = [None for _ in range(2)]
        #TODO heartbeat pipe? ##static pipe

        self.CLUSTER_progress = np.zeros(self.num_workers)
        self.SPADE_progress = np.zeros(self.SPADE_num_workers)

        self.running = 0

        self.centroid_invar = None
        self.centroid_outvar = None
        self.centroid_lambda_weighting = None
        self.centroid_level_mapping = None
        self.centroid_values = None
        self.prev_centroid_values = None
        self.centroid_keys = None
        self.prev_centroid_keys = None
        self.centroid_sizes = None
        self.prev_centroid_sizes = None
        self.cluster_subsets = None
        self.prev_cluster_subsets = None

        self.hitCount=1
        self.missCount=1

    def start_SPADE_Workers(self, detail, dataset_input, dataset_output):
        for wID in range(self.SPADE_num_workers):
            pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
            worker = mp.Process(
                target=worker_SPADE_fn, args=(wID, detail,
                                            dataset_input[wID], dataset_output[wID],
                                            pipe_fromThread_send, 
                                            self.knn, self.num_eigs,
                                            self.initial_graph_vars, 
                                            self.SPADE_vars)
            )
            worker.daemon = True
            worker.start()
            self.SPADE_workers[wID] = worker
            self.SPADE_pipes_recv[wID] = pipe_fromThread_recv

    def start_CLUSTER_Workers(self, detail, dataset):
        for wID in range(self.num_workers):
            pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
            worker = mp.Process(
                target=worker_GraphZoom_fn, args=(wID, detail,
                                            dataset[wID],
                                            pipe_fromThread_send, 
                                            self.knn,
                                            self.level, self.r, 
                                            self.initial_graph_vars,
                                            self.graph_vars)
            )
            worker.daemon = True
            worker.start()
            self.CLUSTER_workers[wID] = worker
            self.CLUSTER_pipes_recv[wID] = pipe_fromThread_recv
    
    def start_BATCH_worker(self, wID, embeddings):
        if self.BATCH_workers[0]:
            print(f'BATCH WORKERS RESETTING: {self.BATCH_workers[0]}')
            self.BATCH_workers[0].kill() #there won't be another attempt to read until this fcn is done and overwrites the pipes, 
                                           #this should be OK
        #if not self.BATCH_workers[]:
            #raise NotImplementedError
        #if not self.BATCH_workers[1]:
        #    print(f'MAKING NEW WORKER: {self.BATCH_workers[0]}')
        pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
        pipe_fromMain_recv, pipe_fromMain_send = mp.Pipe()
        worker = mp.Process(
            target=worker_BATCH_fn, args=(wID,
                                        embeddings,
                                        self.batch_size,
                                        pipe_fromThread_send, 
                                        pipe_fromMain_recv)
        )
        worker.daemon = True
        worker.start()
        self.BATCH_workers[0] = worker
        #self.BATCH_workers[1] = worker
        print(f'BATCH WORKERS SET: {self.BATCH_workers[0]}')
        self.BATCH_pipes_recv[0] = pipe_fromThread_recv
        self.BATCH_pipes_recv[1] = pipe_fromMain_send

    # def start_BATCH_worker_OLD(self, wID):
    #     if self.BATCH_workers[0]:
    #         print(f'BATCH WORKERS RESETTING: {self.BATCH_workers[0]}')
    #         self.BATCH_workers[0].kill() #there won't be another attempt to read until this fcn is done and overwrites the pipes, 
    #                                        #this should be OK
    #     #if not self.BATCH_workers[]:
    #         #raise NotImplementedError
    #     #if not self.BATCH_workers[1]:
    #     #    print(f'MAKING NEW WORKER: {self.BATCH_workers[0]}')
    #     pipe_fromThread_recv, pipe_fromThread_send = mp.Pipe()
    #     pipe_fromMain_recv, pipe_fromMain_send = mp.Pipe()
    #     worker = mp.Process(
    #         target=worker_BATCH_fn, args=(wID,
    #                                     self.centroid_values, 
    #                                     self.SPADE_combined_results,
    #                                     self.CLUSTER_combined_results,
    #                                     pipe_fromThread_send, 
    #                                     pipe_fromMain_recv)
    #     )
    #     worker.daemon = True
    #     worker.start()
    #     self.BATCH_workers[0] = worker
    #     #self.BATCH_workers[1] = worker
    #     print(f'BATCH WORKERS SET: {self.BATCH_workers[0]}')
    #     self.BATCH_pipes_recv[0] = pipe_fromThread_recv
    #     self.BATCH_pipes_recv[1] = pipe_fromMain_send


    def update_dataset(self, dataset, counter):
        self.dataset = dataset
        ##start = time.time()
        ##np.savez(f'./wholeData_in',self.dataset)
        ##end = time.time()
        ##with open('./logging.txt', 'a+') as f:
            ##f.write(f'wholeData_in, {end-start} \n')
        print('UPDATED DATASET')
    
    def update_dataset_outputs(self, dataset, counter):
        if self.dataset_output:
            print('PUSH-BACK LAST DATASET_OUTPUT')
            self.dataset = self.dataset_output
        self.dataset_output = dataset
        start = time.time()
        np.savez(f'./wholeData_out',dataset)
        end = time.time()
        with open('./logging.txt', 'a+') as f:
            f.write(f'Save wholeData_out, {end-start} \n')
        print('UPDATED DATASET_OUTPUT')

    def update_centroid_vars(self, invar, outvar, lambda_weighting, level_mapping, sizes,counter):
        self.centroid_invar = invar
        self.centroid_outvar = outvar
        self.centroid_lambda_weighting = lambda_weighting
        self.centroid_level_mapping = level_mapping
        self.prev_centroid_sizes = self.centroid_sizes
        self.centroid_sizes = sizes
        ##start = time.time()
        ##np.savez(f'./centroids_vars',invar,outvar,lambda_weighting)
        ##np.savez(f'./centroids_vars',level_mapping, sizes)
        ##end = time.time()
        ##with open('./logging.txt', 'a+') as f:
            ##f.write(f'Save centroids_vars, {end-start} \n')
        print('UPDATED centroid_vars')

    def update_centroid_values(self, values, keys, cluster_subsets, counter):
        self.prev_centroid_values = self.centroid_values
        self.prev_centroid_keys = self.centroid_keys
        self.prev_cluster_subsets = self.cluster_subsets
        self.centroid_values = values
        self.centroid_keys = keys
        self.cluster_subsets = cluster_subsets
        ##start = time.time()
        ##np.savez(f'./centroids_values',keys,values)
        ##end = time.time()
        ##with open('./logging.txt', 'a+') as f:
            ##f.write(f'Save centroids_values, {end-start} \n')
        print('UPDATED centroid_values')
    
    def new_clusters(self, detail = None):
        if detail == None:
            start = time.time()
            try:
               os.replace('./Pre-Recluster_1.npz', './Pre-Recluster_0.npz')
            except:
                print('Failed to move Pre-Recluster_1')
            np.savez(f'./Pre-Recluster_1',{'SPADE':self.SPADE_combined_results,
                                              'CLUSTER':self.centroid_values,
                                              'Ps':self.CLUSTER_combined_results,
                                              'centroids':self.centroid_invar,
                                              'cluster_subsets':self.cluster_subsets},
                                              )
            end = time.time()
            with open('./logging.txt', 'a+') as f:
                f.write(f'Save Re-Cluster, {end-start} \n')
        self.binned_datasets_out = [{k:v[subset] for k,v in self.dataset_output.items()} for subset in self.binned_datasets_refs]
        print('RE-BINNED DATASETS')
        print(f'Starting {self.num_workers} CLUSTER Workers, Details: {detail}')
        self.start_CLUSTER_Workers(detail, self.binned_datasets_out)
        print("Launched HyperEF")
        self.running = 1
    
    def new_spade_scores(self, detail = None):
        if self.SPADE_GRID:
            raise NotImplementedError
        else:
            dataset = {key:self.prev_centroid_values[:,idx,None] for idx,key in enumerate(self.prev_centroid_keys) if key in self.SPADE_vars}
            dataset_output = {key:self.centroid_values[:,idx,None] for idx,key in enumerate(self.centroid_keys) if key in self.SPADE_vars}
        print(f'Starting {self.SPADE_num_workers} SPADE Workers, Details: {detail}')
        self.start_SPADE_Workers(detail, [dataset], [dataset_output])
        print("Launched SPADE")
        self.running = 2

    def subset_refresh(self, update_fcn, total, counter,
                        lda = 0.005, power = True,
                        shuffle = True, avg = True): #initiate the subset refresh, don't hang (start SPADE)
        print('REFRESHING SUBSET VALUES')
        print(f'Total: {total}')
        #TODO start SPADE update
        #NOTE switch between center point and subset estimate
        start = time.time()
        #refresh_importance_collect, refresh_importance_vars = update_fcn(self.centroid_invar, self.centroid_level_mapping)
        refresh_importance_collect, refresh_importance_vars, cluster_subsets= update_fcn(
            self.centroid_level_mapping, sample = self.sample_ratio)
        end = time.time()
        with open('./logging.txt', 'a+') as f:
            f.write(f'Subset Refresh, {end-start} \n')
        #if refreshing spade
        #option to also not overwite old, to use original snapshot for SPADE INPUT only.
        self.update_centroid_values(refresh_importance_collect, refresh_importance_vars, cluster_subsets, counter)
        #self.new_spade_scores()
        #else only update losses?    
        #update centroid values w/out overwriting old, so SPADE can use a longer-term change
        start = time.time()
        #only include LOSS_VARS
        use_losses = [(k,v) for k,v in enumerate(refresh_importance_vars) if v in self.LOSS_vars]
        print(f'Losses to use: {use_losses} of {refresh_importance_vars}')
        print(f'Importance Collect:{refresh_importance_collect.shape}')
        refresh_importance_collect = refresh_importance_collect[:,[k[0] for k in use_losses]]
        print(f'Importance Collect:{refresh_importance_collect.shape}')

        #np array of all subsamples, cluster_subsets contains info for summing.
        #map cluster subsampled losses back to their clusters 
        
        idx = 0
        step = 0
        stop = refresh_importance_collect.shape[0] #total number of subsamples
        n_data = np.zeros((self.centroid_level_mapping.shape[1],refresh_importance_collect.shape[1])) #final number of clusters
        while idx < stop:
            stepSize = cluster_subsets[idx][3] #steps to the beginning of the next cluster
            ##TODO any other normalization or accounting for different cluster sizes? (clusterMap[1])
            #n_data[step,:] = np.sum(importance_collect[idx:idx+stepSize], axis = 0) #summing values from this step's cluster
            n_data[step,:] = np.average(refresh_importance_collect[idx:idx+stepSize], axis = 0) #summing values from this step's cluster
            # if False:
                # n_data[step] = n_data[step]/cluster_subsets[idx][1] #avg by # samples in cluster
                # n_data[step] = n_data[step]/(stepSize) #simple avg
                # n_data[step] = n_data[step]*(cluster_subsets[idx][1]/stop) #scale by cluster size relative to total
            idx = idx + stepSize #move idx to next cluster
            step = step + 1 #iterate step
        assert (n_data.shape[0] == self.centroid_level_mapping.shape[1] 
                and n_data.shape[1] == refresh_importance_collect.shape[1])

        ###self.SPADE_combined_results,
        embeddings = np.linalg.norm(n_data, axis=1)[:,None] + 10 
        # embeddings = refresh_importance_collect
        # embeddings = np.abs(embeddings) #ensure positive
        # embeddings = preprocessing.scale(embeddings,with_mean=False)
        # embeddings = np.sum(embeddings, axis=1)[:,None] #sum losses of each sample
        # np.savez('cluster_scores',embeddings)
        # coarse_LOSS = preprocessing.normalize(coarse_LOSS, axis=0)**2
        projections = self.CLUSTER_combined_results[0]
        laplacians = self.CLUSTER_combined_results[1]
        level = self.CLUSTER_combined_results[2]
        ######Refinement######
        print("%%%%%% Starting Graph Refinement %%%%%%")
        refine_start = time.process_time()
        embeddings   = GzUtils.refinement(level, projections, laplacians, embeddings,
                        lda = lda, power = power,
                        ) #TODO lda/power??
        refine_time  = time.process_time() - refine_start
        print(f"%%%%%% Refine time:{refine_time} %%%%%%")
        end = time.time()
        print(f'TIME TAKEN {end-start}', flush = True)
        with open('./logging.txt', 'a+') as f:
            f.write(f'Batch Score Refinement, {end-start} \n')
        #scores = np.sum(embeddings, axis=1)[:,None]
        ##start = time.time()
        ##np.savez('sample_scores',embeddings)
        ##end = time.time()
        ##with open('./logging.txt', 'a+') as f:
            ##f.write(f'Save sample_scores, {end-start} \n')
        print(f"%%%%%% SCORES: {embeddings.shape} %%%%%%")
        start = time.time()
        #try:
           #os.replace('./Cluster_Raw_1.npz', './Cluster_Raw_0.npz')
        #except:
           #print('Failed to move Cluster_Raw_1')
        #np.savez(f'./Cluster_Raw_1',{'SPADE':self.SPADE_combined_results,
                                              #'CLUSTER':self.centroid_values,
                                              #'Ps':self.CLUSTER_combined_results,
                                              #'centroids':self.centroid_invar,
                                              #'cluster_subsets':cluster_subsets,
                                              #'n_data':n_data,
                                              #'embeddings':embeddings},
                                              #)
        end = time.time()
        with open('./logging.txt', 'a+') as f:
            f.write(f'Save Cluster_Raw, {end-start} \n')
        self.start_BATCH_worker(0,embeddings)
        
    def batch_receiver(self, batch_iterations):
        #TODO fix
        print(f'Hits:{self.hitCount}, \
              misses:{self.missCount}, \
              {self.hitCount/(self.missCount+self.hitCount)}')
        #while True:
        print('CHECKING FOR BATCH')
        print(f'{self.BATCH_pipes_recv[0]}')
        if self.BATCH_pipes_recv[0].poll(timeout = .1):
            message = self.BATCH_pipes_recv[0].recv()
            wID = message[0]
            print(f"Recieved Batch from {wID}")
            print(type(message[1]))
            print(f'pipe: {self.BATCH_pipes_recv}')
            self.hitCount+=message[1][0].shape[0]//self.batch_size
            return message[1][0], False, message[1][1] ##probs
        else:
            self.missCount+=100
            print(f'Batch reciever no data', flush = True)
            current_dataset_size = list(self.dataset.values())[0].shape[0]
            #return np.random.permutation(range(current_dataset_size))
            return np.random.choice(range(current_dataset_size),
                                    size=int(self.batch_size*100),
                                    replace=False)[:,None], True, None
        # if self.BATCH_workers[1] is NONE:
        #     continue
        # else if self.BATCH_pipes_recv[X].poll(timeout = .3): #TODO revise naming, organize hb and data queues
        #     message = self.BATCH_pipes_recv[X].recv()
        #     wID = message[0]
        #     print(f"Recieved Batch from {wID}")
        #     print(type(message[1]))
        #     print(f'pipe: {self.BATCH_pipes_recv}')
        #     self.hitCount+=message[1].shape[0]//self.batch_size
        #     print('Killing old worker')
        #     if self.BATCH_workers[0]:
        #           print(f'BATCH WORKERS RESETTING: {self.BATCH_workers[0]}')
        #           self.BATCH_workers[0].kill() #there won't be another attempt to read until this fcn is done and overwrites the pipes, 
        #           #TODO move pipes/queues
        #           #self.BATCH_workers[0] = self.BATCH_workers[1]
        #           #self.BATCH_workers[1] = None
        #     return message[1], False
        # print('BATCH PREV WORKER')
        # if self.BATCH_pipes_recv[0].poll(timeout = .3):
        #     message = self.BATCH_pipes_recv[0].recv()
        #     wID = message[0]
        #     print(f"Recieved Batch from {wID}")
        #     print(type(message[1]))
        #     print(f'pipe: {self.BATCH_pipes_recv}')
        #     self.hitCount+=message[1].shape[0]//self.batch_size
        #     return message[1], False
        # if self.BATCH_pipes_recv[0].poll(timeout = .3):
        #     self.missCount+=100
        #     print(f'Batch reciever no data', flush = True)
        #     current_dataset_size = list(self.dataset.values())[0].shape[0]
        #     #return np.random.permutation(range(current_dataset_size))
        #     return np.random.choice(range(current_dataset_size),size=int(self.batch_size*100),replace=False)[:,None], True
    
    def check_graph_clustering(self, counter, level = None, detail = None):
        if not level:
            level = self.level
            print(f'Default Level:{level}')
        if self.running == 1:
            print('checking CLUSTERING queues')
            print(f'CLUSTER_progress: {self.CLUSTER_progress}')
            if not (all(self.CLUSTER_progress == 1)):
                for i in range(self.num_workers): #limit no. of checks (don't while loop) so training can continue while waiting
                    if self.CLUSTER_progress[i] == 1:
                        print(f'Cluster {i} Recieved, waiting')
                        continue ##check next
                    if self.CLUSTER_pipes_recv[i].poll(timeout = .1):
                        message = self.CLUSTER_pipes_recv[i].recv()
                        wID = i
                        print(f"Recieved Clusters for {wID}")
                        print(type(message[1]))
                        self.CLUSTER_binned_results[wID] = message[1]
                        self.CLUSTER_progress[wID] = 1
                    else:
                        print(f'Cluster attempt {i} NoMsg, ', flush = True)
                        continue ##no message, check next
            print(f'CLUSTER_progress: {self.CLUSTER_progress}')
            if not (all(self.CLUSTER_progress == 1)):
                return False
            print(f'CLUSTER_progress: {self.CLUSTER_progress}')
            ##this differs between sim_coarse and HyperEF, TODO handle
            ##Projections, Laplacians, Level vs idx_list
            self.CLUSTER_combined_results = self.combine_grids_clusters(self.CLUSTER_binned_results, 
                                                                 self.binned_datasets_refs,
                                                                 level)
            ##self.centroid_level_mapping = idxLists_to_matrix_level(self.CLUSTER_combined_results,level)
            self.centroid_level_mapping = reduce(lambda x,y: x@y, self.CLUSTER_combined_results[0])
            self.CLUSTER_progress = np.zeros(self.num_workers) #RESET
            self.running = False
            return True
        else:
            print(f'Graph not running, skipping check.')
            return False
        
    def check_SPADE_progress(self,counter,detail=None): #replace batch worker when done
        if not self.running == 2:
            print(f'SPADE not running, skipping check.')
            return False
        print(f'SPADE_progress: {self.SPADE_progress}')
        if not (all(self.SPADE_progress == 1)):
            for i in range(self.SPADE_num_workers): #limit no. of checks (don't while loop) so training can continue while waiting
                if self.SPADE_progress[i] == 1:
                    print(f'SPADE grid {i} Recieved, waiting') #cont to CLUSTER
                else:
                    if self.SPADE_pipes_recv[i].poll(timeout = .1):
                        wID = i
                        print(f"Recieved SPADE for {wID}")
                        message = self.SPADE_pipes_recv[i].recv()
                        print(type(message[1]))
                        ##TopEig = message[1][0]
                        ##TopEdgeList = message[1][1]
                        ##TopNodeList = message[1][2]
                        node_score = message[1][3]
                        self.SPADE_binned_results[wID] = node_score
                        self.SPADE_progress[wID] = 1
                    else:
                        print(f'SPADE attempt {i} NoMsg, ', flush = True)
        print(f'SPADE_progress: {self.SPADE_progress}')
        if not (all(self.SPADE_progress == 1)):
            return False 
        print('SPADE COMPLETE/RESET')
        self.SPADE_progress = np.zeros(self.SPADE_num_workers) #RESET
        self.running = False
        self.SPADE_combined_results = self.combine_grids_nodes(self.SPADE_binned_results, self.binned_datasets_refs) #combine SPADE grids
        #SPADE_combined_results and centroid_values have in-order per-cluster SPADE SCORE and LOSSES respectively
        #kills current batch worker and starts a new one with latest results
        #try:
        #    os.replace('./Cluster_Raw_1.npz', './Cluster_Raw_0.npz')
        #except:
        #    print('Failed to move Cluster_Raw_1')
        #np.savez(f'./Cluster_Raw_1',{'SPADE':self.SPADE_combined_results,
        #                                       'CLUSTER':self.centroid_values,
        #                                       'Ps':self.CLUSTER_combined_results,
        #                                       'centroids':self.centroid_invar})
        self.start_BATCH_worker(0)
        #np.savez(f'./combinedGrids_{counter}',{'SPADE':self.SPADE_combined_results,
        #                                       'CLUSTER':self.CLUSTER_combined_results,
        #                                       'dataset_in':self.dataset,
        #                                      'dataset_out':self.dataset_output})
        # with open('./logging.txt', 'a+') as f:
        #         counts = np.array([self.CLUSTER_combined_results.indptr[i+1]-self.CLUSTER_combined_results.indptr[i]
        #                   for i in range(0,self.CLUSTER_combined_results.shape[1])])
        #         #lend = time.time()
        #         f.write(f'P shape:{self.CLUSTER_combined_results.shape},SPADE shape:{self.SPADE_combined_results.shape} \n')
        #         f.write(f'Min: {np.min(counts)}, Max {np.max(counts)}, Mean {np.mean(counts)}, Median {np.median(counts)} \n')
        #         bins = range(np.min(counts),np.max(counts),max(1,int(np.max(counts)/10)))
        #         histogram = np.histogram(counts,bins)
        #         f.write(f'Histogram: {histogram[0]},{histogram[1]}\n')
        #        #lstart = time.time()
        #for i in range(len(self.binned_datasets_refs)):
            #np.savez(f'./Step{counter}_GridRefs_{i}',np.array(['Interior',self.binned_datasets_refs[i]],dtype='object'))
            #np.savez(f'./Step{counter}_GridIn_{i}',np.array(['Interior',self.binned_datasets_in[i]],dtype='object'))
            #np.savez(f'./Step{counter}_GridOut_{i}',np.array(['Interior',self.binned_datasets_out[i]],dtype='object'))
            #np.savez(f'./Step{counter}_GridSPADE_{i}',np.array(['Interior',self.SPADE_binned_results[i]],dtype='object'))
        return True

    @staticmethod
    def combine_grids_nodes(binned_node_scores, binned_refs): #combine the spade scores of all nodes in all grids in original order.
        print(f'COMBINEDGRIDSNODES{len(binned_node_scores)}')
        if len(binned_node_scores) == 1:
            return SPADE_uniform_quantile(binned_node_scores[0])
        raise NotImplementedError
        grid_node_scores = np.concatenate(binned_node_scores, axis=0) #all scores
        refs_all_grids = np.concatenate(binned_refs, axis=0) #a list of n indexes, in the same order as the scores, with the index for that score in the original dataset
        return SPADE_uniform_quantile(grid_node_scores[np.argsort(refs_all_grids)])
    
    @staticmethod
    def combine_grids_clusters(binned_clusters,binned_refs,level): #combine the direct L outputs of new_graph
        if len(binned_clusters) ==  1:
            return binned_clusters[0]
        #np.savez('grid-separated-clusters',binned_clusters,binned_refs)
        levels = [i[2] for i in binned_clusters]
        binned_laps = [i[1] for i in binned_clusters]
        binned_proj = [i[0] for i in binned_clusters]
        assert np.all(np.array(levels)==levels[0])
        levels = levels[0]
        Projections = []
        Laplacians = []
        print(levels)
        for level in range(levels): #mappings
            #Proj
            print(f'PROJECTION {level}')
            Cn = [bin[level] for bin in binned_proj]
            if level == 0:
                proj = combineGrids_helper(Cn, binned_refs, 'proj')
            else:
                proj = combineGrids_helper(Cn)
            Projections.append(proj)
            #Lap
            print(f'LAPLACIAN {level}')
            Cn = [bin[level] for bin in binned_laps]
            if level == 0:
                lap = combineGrids_helper(Cn, binned_refs, 'lap')
            else:
                lap = combineGrids_helper(Cn)
            Laplacians.append(lap)
        return [Projections, Laplacians, levels]
    
    @staticmethod
    def split_grids(dataset, grid_width):
        values = {}
        ranges = {}
        dkeys = [i for i in ['x','y','z'] if i in dataset.keys()]
        if dkeys == []:
            raise ValueError('No x,y,z data')
        for cord in dkeys:
            values[cord] = dataset[cord].cpu().detach().numpy()
            ranges[cord] = (np.min(dataset[cord].cpu().detach().numpy()),
                            np.max(dataset[cord].cpu().detach().numpy()),
                            np.ptp(dataset[cord].cpu().detach().numpy()))
        #print('ranges', ranges)
        for k,cord in enumerate(dkeys):
            #print(cord)
            values[cord] = values[cord] - ranges[cord][0] # shift to start at 0
            #print(values[cord])
            #print((values[cord]/((ranges[cord][2]/grid_width)+1e-8)).astype(int))
            if k == 1: #y goes 'down', i.e. topmost row is y=0 and bottom-most row is y=grid_width-1
                values[cord] = np.clip(grid_width - 1 - (values[cord]/((ranges[cord][2]/grid_width))).astype(int), 0, grid_width-1) # grid coord of each point
            else:
                values[cord] = np.clip((values[cord]/((ranges[cord][2]/grid_width))).astype(int), 0, grid_width-1) # grid coord of each point
        grids = np.zeros((values['x'].shape[0]),dtype=np.uint32) #there should always be at least x
        #x+grid_width*y+grid_width^2*z is the box number
        for k, cord in enumerate(dkeys):
            grids = np.add(grids,np.multiply(values[cord].reshape(values[cord].shape[0]),int(grid_width**k)))
        #print(grids)
        blocks = [[] for _ in range(0,grid_width**len(dkeys))]
        for d,b in enumerate(grids):
            blocks[b].append(d)
        return [np.array(i) for i in blocks]
    
if __name__ == '__main__':
    #main()
    pass

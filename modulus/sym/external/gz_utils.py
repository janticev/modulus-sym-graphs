import numpy as np
from numpy import linalg as LA
import json
import networkx as nx
from networkx.readwrite import json_graph
from networkx.linalg.laplacianmatrix import laplacian_matrix
from scipy.io import mmwrite
from scipy.sparse import csr_matrix, diags, identity, triu, tril
from itertools import combinations

import cupyx.scipy as cps
import cupy as cp

import os
import sys
from sklearn.preprocessing import normalize
import time
import pdb

import cupy as cp
import cupyx.scipy as cps
def refinement(levels, projections, coarse_laplacians, embeddings, lda = 0.1, 
               power = False):
    #pdb.set_trace()
    embeddings_gpu = cp.asarray(embeddings)
    #print(type(embeddings_gpu))
    #print(embeddings_gpu.shape)
    projections_gpu = [cps.sparse.csr_matrix(i.tocsr(copy=True)) for i in projections]
    for i in reversed(range(levels)):
        print(i)
        #print('embeddings',embeddings_gpu.shape)
        #print('projections', projections_gpu[i].shape)
        embeddings_gpu = projections_gpu[i].dot(embeddings_gpu)
        #print('embeddings',embeddings_gpu.shape)
        start = time.time()
        #print(type(coarse_laplacians[i]))
        filter_    = smooth_filter_gpu(coarse_laplacians[i], lda)
        #filter_    = cps.sparse.csr_matrix(smooth_filter(laplacians[i], lda))
        #print('filter',type(filter_),filter_.shape)
        end = time.time()
        #print(f'Filter Time:{end-start}')
        ## power controls whether smoothing intermediate embeddings,
        ## preventing over-smoothing
        if power or i == 0:
            #embeddings_gpu = filter_ @ (filter_ @ embeddings_gpu)
            #embeddings_gpu = filter_.dot(filter_.dot(embeddings_gpu))       #2
            #embeddings_gpu = filter_.dot(filter_.dot(filter_.dot(embeddings_gpu)))        #3
            #embeddings_gpu = filter_.dot(filter_.dot(filter_.dot(filter_.dot(embeddings_gpu))))        #4
            embeddings_gpu = filter_.dot(filter_.dot(filter_.dot(filter_.dot(filter_.dot(embeddings_gpu)))))        #5
    embeddings = embeddings_gpu.get()
    return embeddings

def smooth_filter_gpu(laplacian_matrix, lda):
    start = time.time()
    dim        = laplacian_matrix.shape[0]
    adj_matrix = (cps.sparse.diags(laplacian_matrix.diagonal(), 0, format='csr') - 
                    cps.sparse.csc_matrix(laplacian_matrix) + 
                    lda * cps.sparse.identity(dim,format='csr')
                )
    end = time.time()
    #print('adj',end-start)
    #print('adj_matrix',type(adj_matrix),adj_matrix.shape)
    start = time.time()
    degree_vec = adj_matrix.sum(axis=1)
    end = time.time()
    #print('deg_vec',end-start)
    #print('degree_vec',type(degree_vec),degree_vec.shape)
    start = time.time()
    with np.errstate(divide='ignore'):
        d_inv_sqrt = cp.squeeze(cp.asarray(cp.power(degree_vec, -0.5)))
    end = time.time()
    #print('d_inv_1',end-start)
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    end = time.time()
    #print('d_inv_2',end-start)
    start = time.time()
    degree_matrix  = cps.sparse.diags(d_inv_sqrt, 0, format='csr')
    end = time.time()
    #print('degree_matrix',end-start)
    #print('degree_matrix',type(degree_matrix),degree_matrix.shape)
    #norm_adj       = degree_matrix @ (adj_matrix @ degree_matrix)
    norm_adj       = degree_matrix.dot(adj_matrix.dot(degree_matrix))
    #norm_adj       = degree_matrix.multiply(adj_matrix.multiply(degree_matrix))

    end = time.time()
    #print('norm_adj',end-start)
    return norm_adj


####CPU-ONLY REFINEMENT AND FILTER
# def refinement(levels, projections, coarse_laplacian, embeddings, 
#                lda = 0.1, 
#                power = False):
#     for i in reversed(range(levels)):
#         embeddings = projections[i] @ embeddings
#         filter_    = smooth_filter(coarse_laplacian[i], lda)

#         ## power controls whether smoothing intermediate embeddings,
#         ## preventing over-smoothing
#         if power or i == 0:
#             embeddings = filter_ @ (filter_ @ embeddings)
#     return embeddings
#
def smooth_filter(laplacian_matrix, lda):
    dim        = laplacian_matrix.shape[0]
    adj_matrix = diags(laplacian_matrix.diagonal(), 0) - laplacian_matrix + lda * identity(dim)
    degree_vec = adj_matrix.sum(axis=1)
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.squeeze(np.asarray(np.power(degree_vec, -0.5)))
    d_inv_sqrt[np.isinf(d_inv_sqrt)|np.isnan(d_inv_sqrt)] = 0
    degree_matrix  = diags(d_inv_sqrt, 0)
    norm_adj       = degree_matrix @ (adj_matrix @ degree_matrix)
    return norm_adj

# ######Graph Reduction######
# print("%%%%%% Starting Graph Reduction %%%%%%")
# reduce_start = time.process_time()
# G, projections, laplacians, level = sim_coarse(laplacian, args.level)
# reduce_time = time.process_time() - reduce_start

# ######Refinement######
# print("%%%%%% Starting Graph Refinement %%%%%%")
# refine_start = time.process_time()
# embeddings   = refinement(level, projections, laplacians, embeddings, args.lda, args.power)
# refine_time  = time.process_time() - refine_start


######Save Embeddings######
#np.save(args.embed_path, embeddings)


######Evaluation######
#lr("dataset/{}/".format(dataset), args.embed_path, dataset)

#if __name__ == "__main__":
    #sys.exit(main())

def affinity(x, y):
    dot_xy = (np.dot(x, y))**2
    norm_x = (LA.norm(x))**2
    norm_y = (LA.norm(y))**2
    return dot_xy/(norm_x*norm_y)

def spec_coarsen(filter_, laplacian):
    np.random.seed(seed=1)

    ## power of low-pass filter
    power = 2
    ## number of testing vectors
    t = 7
    ## threshold for merging nodes
    thresh = 0.3

    adjacency = diags(laplacian.diagonal(), 0) - laplacian
    G = nx.from_scipy_sparse_matrix(adjacency)
    tv_list = []
    num_nodes = len(G.nodes())

    ## generate testing vectors in [-1,1], 
    ## and orthogonal to constant vector
    for _ in range(t):
        tv = -1 + 2 * np.random.rand(num_nodes)
        tv -= np.ones(num_nodes)*np.sum(tv)/num_nodes
        tv_list.append(tv)
    tv_feat = np.transpose(np.asarray(tv_list))

    ## smooth the testing vectors
    for _ in range(power):
        tv_feat = filter_ @ tv_feat
    matched = [False] * num_nodes
    degree_map = [0] * num_nodes

    ## hub nodes are more important than others,
    ## treat hub nodes as seeds
    for (node, val) in G.degree():
        degree_map[node] = val
    sorted_idx = np.argsort(np.asarray(degree_map))
    row = []
    col = []
    data = []
    cnt = 0
    for idx in sorted_idx:
        if matched[idx]:
            continue
        matched[idx] = True
        cluster = [idx]
        for n in G.neighbors(idx):
            if affinity(tv_feat[idx], tv_feat[n]) > thresh and not matched[n]:
                cluster.append(n)
                matched[n] = True
        row += cluster
        col += [cnt] * len(cluster)
        data += [1] * len(cluster)
        cnt += 1
    mapping = csr_matrix((data, (row, col)), shape=(num_nodes, cnt))
    coarse_laplacian = mapping.transpose() @ laplacian @ mapping
    return coarse_laplacian, mapping

def sim_coarse(laplacian, level):
    projections = []
    laplacians = []
    for i in range(level):
        filter_ = smooth_filter(laplacian, 0.1)
        laplacians.append(laplacian)
        laplacian, mapping = spec_coarsen(filter_, laplacian)
        projections.append(mapping)

        print("Coarsening Level:", i+1)
        print("Num of nodes: ", laplacian.shape[0], "Num of edges: ", int((laplacian.nnz - laplacian.shape[0])/2))

    #adjacency = diags(laplacian.diagonal(), 0) - laplacian
    #G = nx.from_scipy_sparse_matrix(adjacency, edge_attribute='wgt')
    return projections, laplacians, level#, G
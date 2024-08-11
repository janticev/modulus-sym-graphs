# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Modulus Dataset constructors for continuous type data
"""

from typing import Dict, List, Callable, Union

import numpy as np

from modulus.sym.utils.io.vtk import var_to_polyvtk
from .dataset import Dataset, IterableDataset, _DictDatasetMixin

from scipy.sparse import csr_matrix, csc_matrix, diags, identity, triu, tril

import pdb
from sklearn import preprocessing

#from torch.utils.tensorboard import SummaryWriter

from modulus.external.multithreading import MultiProcessGZ

import pdb as pdb
import multiprocessing as mp
from multiprocessing.queues import Empty
import time
import torch

class _DictPointwiseDatasetMixin(_DictDatasetMixin):
    "Special mixin class for dealing with dictionaries as input"

    def save_dataset(self, filename):

        named_lambda_weighting = {
            "lambda_" + key: value for key, value in self.lambda_weighting.items()
        }
        save_var = {**self.invar, **self.outvar, **named_lambda_weighting}
        var_to_polyvtk(filename, save_var)


class DictPointwiseDataset(_DictPointwiseDatasetMixin, Dataset):
    """A map-style dataset for a finite set of pointwise training examples."""

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        lambda_weighting: Dict[str, np.array] = None,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length


class DictInferencePointwiseDataset(Dataset):
    """
    A map-style dataset for inferencing the model, only contains inputs
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        output_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.output_names = output_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return (invar,)

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.output_names)


class ContinuousPointwiseIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of pointwise training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        lambda_weighting_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }

        def iterable_function():
            while True:
                invar = Dataset._to_tensor_dict(self.invar_fn())
                outvar = Dataset._to_tensor_dict(self.outvar_fn(invar))
                lambda_weighting = Dataset._to_tensor_dict(
                    self.lambda_weighting_fn(invar, outvar)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        invar = self.invar_fn()
        return list(invar.keys())

    @property
    def outvar_keys(self):
        invar = self.invar_fn()
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):
    """
    An infinitely iterable dataset that applies importance sampling for faster more accurate monte carlo integration
    """
    def dict_to_numpy(self, invar):
        data = np.concatenate([invar[x] for x in invar.keys()], axis=1)
        num_elements = list(invar.values())[0].shape[0]
        ids = np.arange(num_elements)
        lkeys = invar.keys()
        dim = len(lkeys)
        return data, ids, lkeys, num_elements, dim
    
    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,
        seednum: int = 1000,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure
        self.seednum = seednum

        import time

        def iterable_function():

            # TODO: re-write idx calculation using pytorch sampling - to improve performance

            counter = 0
            btime = []
            batchtime = []
            batchstart = time.time()
            ###############################
            ###############################
            ##Tesselation##
            ###############################
            ###############################
            initial_vars = ['x','y']
            seed_vars = {k:v for k,v in self.invar.items() if k in initial_vars}
            data, ids, lkeys, num_elements, dim = self.dict_to_numpy(seed_vars)
            print(f'INITIAL VARS: {initial_vars}, NUMBER OF SEEDS: {self.seednum}, RATIO: {self.seednum/num_elements}')
            with open('./logging.txt', 'a+') as f:
                    f.write(f'INITIAL VARS: {initial_vars}, NUMBER OF SEEDS: {self.seednum}, RATIO: {self.seednum/num_elements}')
            print('Keys:' + str(lkeys))
            print(f'Size:({num_elements}x{dim})')
            print('Norm')
            data_output = preprocessing.scale(data)
            #seeds and kclusters[i][0] are in the same order.
            seeds = np.random.choice(num_elements, self.seednum, replace=False)
            kclusters = [[i] for i in seeds]
            kCoords = data_output[seeds]
            Sseeds = set(seeds)
            count = 0
            for i in range(0,num_elements):
                count += 1
                if count%(num_elements/10)==0:
                    print(f'{100*(count/num_elements)}')
                if i in Sseeds:
                    continue
                seedIdx = np.argmin(np.linalg.norm(data_output[i]-kCoords,axis=1))
                kclusters[seedIdx].append(i)

            sizes = np.array([len(i) for i in kclusters])
            print(f'Cells: {len(sizes)}')
            print(f'Avg: {np.average(sizes)}')
            print(f'Max: {np.max(sizes)}')
            print(f'Min: {np.min(sizes)}')
            print(f'.75<x: {np.count_nonzero(sizes>self.seednum*.75)}')
            print(f'x<1.25: {np.count_nonzero(sizes<self.seednum*1.25)}')

            seed_invar = {k:v[seeds] for k,v in self.invar.items()}

            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    lstart = time.time()
                    print('RESAMPLE')
                    print(counter)
                    print(self.resample_freq)
                    list_importance = []
                    list_invar = { ### variable: list of batches 
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in seed_invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))): ###getting number of elements in a batch, should all be the same
                        importance = self.importance_measure(  ##per batch importance measure, returns an array the same size as the batch
                            {key: value[i] for key, value in list_invar.items()} ###variable: batch i
                        )
                        list_importance.append(importance) ###list of arrays of importances of each batch
                    importance = np.concatenate(list_importance, axis=0) ###reform list of lists into a Bsize*Bnum by 1 vector, previous batching was to keep calc on GPU?    

                    pw_manual = np.zeros((num_elements,1))
                    for k,i in enumerate(kclusters):
                        pw_manual[i] = np.linalg.norm(importance[k])

                    prob = pw_manual / np.sum(self.invar["area"].numpy() * pw_manual) ###np array of probabilities for each batch? importance/(area*importance of each sample, summed)
                    #probability is importance of a sample divided by area-weighted importance of the batch
                    lend = time.time()
                    with open('./logging.txt', 'a+') as f:
                        f.write(f'average batch time: {np.average(btime)}, total batch time: {np.sum(btime)}')
                        f.write(f',resampled all points, {lend-lstart}, step:{counter}  \n')
                        btime = []

                
                if counter%50000 == 0:
                    start = time.time()
                    list_importance = []
                    list_invar = { ### variable: list of batches 
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in self.invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))): ###getting number of elements in a batch, should all be the same
                        importance = self.importance_measure(  ##per batch importance measure, returns an array the same size as the batch
                            {key: value[i] for key, value in list_invar.items()} ###variable: batch i
                        )
                        list_importance.append(importance) ###list of arrays of importances of each batch
                    importance = np.concatenate(list_importance, axis=0) ###reform list of lists into a Bsize*Bnum by 1 vector, previous batching was to keep calc on GPU? 
                    np.savez('probs_all',prob, self.invar, pw_manual, importance)
                    end = time.time()
                    with open('./logging.txt', 'a+') as f:
                        f.write(f'save probs_all: {end-start}\n')

                # sample points from probability distribution and store idx
                idx = np.array([])
                bstart = time.time()

                idx_size = self.length//16
                div_size = idx_size//4
                for i in range(int(idx_size/div_size)):
                    idx = np.concatenate([idx, np.random.choice(self.length, div_size, replace=False, 
                                        p = preprocessing.normalize(
                                                prob,
                                                norm='l2', 
                                                axis=0, copy=True, return_norm=False
                                             ).reshape(-1,)**2)])
                    
                ## Common    
                idx = idx.astype(np.int64)   
                bend = time.time()
                #print(btime)

                #btime.append((bend-bstart))       #old (toggle)
                btime.append((bend-bstart)/(idx_size//self.batch_size))        #est.  (toggle)                  

                for i in range(len(idx)//self.batch_size):
                    # gather invar, outvar, and lambda weighting
                    idxStart = i*batch_size
                    idxEnd = idxStart+batch_size
                    invar = _DictDatasetMixin._idx_var(self.invar, idx[idxStart:idxEnd])
                    outvar = _DictDatasetMixin._idx_var(self.outvar, idx[idxStart:idxEnd])
                    lambda_weighting = _DictDatasetMixin._idx_var(
                        self.lambda_weighting, idx[idxStart:idxEnd]
                    )

                    # set area value from importance sampling
                    invar["area"] = 1.0 / (prob[idx[idxStart:idxEnd]] * self.batch_size)

                    # return and count up
                    counter += 1
                    if counter % self.resample_freq == 0:
                        break

                    batchend = time.time()
                    batchtime.append((batchend-batchstart))
                    if (len(batchtime) > 1 and batchtime[-1] - batchtime[-2] > 1e-2):
                        with open('./batch_logs.txt', 'a+') as f:
                            f.write(f'----Average batch time: {np.average(batchtime[0:-1])}, total batch time: {np.sum(batchtime[0:-1])},'
                                    + f'step:{counter} to step:{counter-len(batchtime)}\n')
                            f.write(f'----Time since last batch:, {batchend-batchstart}, step:{counter}  \n')
                        batchtime = []
                    batchstart = time.time()
                    yield (invar, outvar, lambda_weighting)

            ###############################
            ###############################
            ##End Tesselation##
            ###############################
            ###############################

            ###############################
            ###############################
            ##Approx Batching##
            ###############################
            ###############################
            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    lstart = time.time()
                    print('RESAMPLE')
                    print(counter)
                    print(self.resample_freq)
                    list_importance = []
                    list_invar = { ### variable: list of batches 
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in self.invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))): ###getting number of elements in a batch, should all be the same
                        importance = self.importance_measure(  ##per batch importance measure, returns an array the same size as the batch
                            {key: value[i] for key, value in list_invar.items()} ###variable: batch i
                        )
                        list_importance.append(importance) ###list of arrays of importances of each batch
                    importance = np.concatenate(list_importance, axis=0) ###reform list of lists into a Bsize*Bnum by 1 vector, previous batching was to keep calc on GPU?
                    prob = importance / np.sum(self.invar["area"].numpy() * importance) ###np array of probabilities for each batch? importance/(area*importance of each sample, summed)
                    #probability is importance of a sample divided by area-weighted importance of the batch
                    start = time.time()
                    np.savez('probs_all',prob, self.invar, importance)
                    end = time.time()
                    lend = time.time()
                    with open('./logging.txt', 'a+') as f:
                        f.write(f'average batch time: {np.average(btime)}, total batch time: {np.sum(btime)}')
                        f.write(f',resampled all points, {lend-lstart}, step:{counter}  \n')
                        f.write(f'save probs_all: {end-start}\n')
                        btime = []
                    
                # sample points from probability distribution and store idx
                idx = np.array([])
                bstart = time.time()

                idx_size = self.length//16
                div_size = idx_size//4
                for i in range(int(idx_size/div_size)):
                    idx = np.concatenate([idx, np.random.choice(self.length, div_size, replace=False, 
                                        p = preprocessing.normalize(
                                                prob,
                                                norm='l2', 
                                                axis=0, copy=True, return_norm=False
                                             ).reshape(-1,)**2)])
                    
                ## Common    
                idx = idx.astype(np.int64)   
                bend = time.time()
                btime.append((bend-bstart)/(idx_size//self.batch_size))           

                for i in range(len(idx)//self.batch_size):
                    # gather invar, outvar, and lambda weighting
                    idxStart = i*batch_size
                    idxEnd = idxStart+batch_size
                    invar = _DictDatasetMixin._idx_var(self.invar, idx[idxStart:idxEnd])
                    outvar = _DictDatasetMixin._idx_var(self.outvar, idx[idxStart:idxEnd])
                    lambda_weighting = _DictDatasetMixin._idx_var(
                        self.lambda_weighting, idx[idxStart:idxEnd]
                    )

                    # set area value from importance sampling
                    invar["area"] = 1.0 / (prob[idx[idxStart:idxEnd]] * self.batch_size)

                    # return and count up
                    counter += 1
                    if counter % self.resample_freq == 0:
                        break

                    batchend = time.time()
                    batchtime.append((batchend-batchstart))
                    if (len(batchtime) > 1 and batchtime[-1] - batchtime[-2] > 1e-2):
                        with open('./batch_logs.txt', 'a+') as f:
                            f.write(f'----Average batch time: {np.average(batchtime[0:-1])}, total batch time: {np.sum(batchtime[0:-1])},'
                                    + f'step:{counter} to step:{counter-len(batchtime)}\n')
                            f.write(f'----Time since last batch:, {batchend-batchstart}, step:{counter}  \n')
                        batchtime = []
                    batchstart = time.time()

                    yield (invar, outvar, lambda_weighting)

            ###############################
            ###############################
            ##Original##
            ###############################
            ###############################

            while True:
                # resample all points when needed
                if counter % self.resample_freq == 0:
                    lstart = time.time()
                    print('RESAMPLE')
                    print(counter)
                    print(self.resample_freq)
                    list_importance = []
                    list_invar = { ### variable: list of batches 
                        key: np.split(value, value.shape[0] // self.batch_size)
                        for key, value in self.invar.items()
                    }
                    for i in range(len(next(iter(list_invar.values())))): ###getting number of elements in a batch, should all be the same
                        importance = self.importance_measure(  ##per batch importance measure, returns an array the same size as the batch
                            {key: value[i] for key, value in list_invar.items()} ###variable: batch i
                        )
                        list_importance.append(importance) ###list of arrays of importances of each batch
                    importance = np.concatenate(list_importance, axis=0) ###reform list of lists into a Bsize*Bnum by 1 vector, previous batching was to keep calc on GPU?
                    prob = importance / np.sum(self.invar["area"].numpy() * importance) ###np array of probabilities for each batch? importance/(area*importance of each sample, summed)
                    #probability is importance of a sample divided by area-weighted importance of the batch
                    start = time.time()
                    np.savez('probs_all',prob, self.invar, importance)
                    end = time.time()
                    lend = time.time()
                    with open('./logging.txt', 'a+') as f:
                        f.write(f'average batch time: {np.average(btime)}, total batch time: {np.sum(btime)}')
                        f.write(f',resampled all points, {lend-lstart}, step:{counter}  \n')
                        f.write(f'save probs_all: {end-start}\n')
                        btime = []
                    
                # sample points from probability distribution and store idx
                idx = np.array([])
                bstart = time.time()

                while True:
                    r = np.random.uniform(0, np.max(prob), size=self.batch_size) ### batch_size array of random vars from 0 to max(prob)
                    try_idx = np.random.choice(self.length, self.batch_size)     ### batch_size array of indexes from dataset
                    if_sample = np.less(r, prob[try_idx, :][:, 0])               ### true/false array of indexes where r < prob
                    idx = np.concatenate([idx, try_idx[if_sample]])              ### add true indexes to the sampling indexes
                    if idx.shape[0] >= batch_size:
                        idx = idx[:batch_size]
                        break

                
                idx = idx.astype(np.int64)   
                bend = time.time()

                btime.append((bend-bstart))   

                # gather invar, outvar, and lambda weighting
                invar = _DictDatasetMixin._idx_var(self.invar, idx)
                outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
                lambda_weighting = _DictDatasetMixin._idx_var(
                    self.lambda_weighting, idx
                )

                # set area value from importance sampling
                invar["area"] = 1.0 / (prob[idx] * batch_size)

                with open('./batch_logs.txt', 'a+') as f:
                    batchend = time.time()
                    f.write(f'----Time since last batch:, {batchend-batchstart}, step:{counter}  \n')
                    batchstart = time.time()

                yield (invar, outvar, lambda_weighting)
                # return and count up
                counter += 1
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

class DictGraphImportanceSampledPointwiseIterableDataset(
    _DictPointwiseDatasetMixin, IterableDataset
):

    """
    An infinitely iterable dataset that implements:
    """

    def warmup_loop(self, indicator_fcn=None,threshold=None,
                    detail = None,
                    message = 'WARMUP SAMPLING'): ##indicator must take self.counter as an input
        if indicator_fcn == None:
            indicator_fcn = lambda x, detail = None: x
        if threshold == None:
            threshold = self.warmup
        indicator = indicator_fcn(self.counter, detail = detail)
        while indicator < threshold: ##warmup before trying to get outputs from NN
            warmup_idx = np.split(np.random.permutation(range(0,self.length)), self.length // self.batch_size)
            print(f'Step: {self.counter}: {message}, Detail:{detail}')
            for i in warmup_idx:
                invar = _DictDatasetMixin._idx_var(self.invar, i)
                outvar = _DictDatasetMixin._idx_var(self.outvar, i)
                lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, i)
                self.counter += 1
                if self.counter%1000 == 0:
                    print("Checking Indicator")
                    indicator = indicator_fcn(self.counter, detail=detail)
                    print(f'Indicator: {indicator}, {threshold}, {indicator<threshold}')
                    if not (indicator < threshold):
                        break
                invar["area"] *= self.batch_per_epoch #area scaling uniform
                yield (invar, outvar, lambda_weighting)
            

    def running_loop(self):
        pass
    
    def importance_update(self):
        ##Get output data as np.
        print(f'Calculating current outputs for clustering...{self.graph_vars} out of {self.invar.keys()}')
        list_invar = {
            key: np.split(value, value.shape[0] // self.batch_size)
            for key, value in _DictDatasetMixin._idx_var(self.invar, range(0,self.length)).items()
        }
        importance_collect = []
        importance_vars = []
        for i in range(len(next(iter(list_invar.values())))):
            importance = self.importance_measure(
                {key: value[i] for key, value in list_invar.items()}
            )
            importance_collect.append(importance[0])
            if not importance_vars:
                importance_vars = importance[1] ##the vars defined in importance function
        return np.concatenate(importance_collect, axis= 0), importance_vars
    
    def importance_update_subsets(self,P,sample = 1):
        start = time.time()
        if not type(P) == csc_matrix:
            P = P.tocsc(copy=False)
        n = P.shape[1]
        print(f'Clusters: {n}, Samples: {P.shape[0]}, sPerCluster: {sample}')
        #columns is an array of the number of elements in each column (samples per cluster)
        columns = np.array([P.indptr[i+1]-P.indptr[i] for i in range(0,P.shape[1])])
        #selections is an array of index choices for each row 
        if sample > 1:
            selectSize = np.minimum(sample,columns)
        elif 0 < sample < 1:
            start = time.time()
            selectSize = np.maximum(1,(columns*sample)).astype(int)
            end = time.time()
            print(end-start)
            print(min(selectSize))
            print(max(selectSize))
            print(max(columns))
        else:
            selectSize = np.ones(len(columns))
        start = time.time()
        try:
            rand = np.random.default_rng() #for later versions of numpy
            selections = [rand.integers(columns[i],size = selectSize[i]) for i in range(0,len(columns))]
        except:
            selections = [np.random.randint(columns[i],size = selectSize[i]) for i in range(0,len(columns))] #for earlier versions of numpy
        end = time.time()
        print(end-start)
        out = np.hstack([np.array([
                    P.indices[P.indptr[i]:P.indptr[i+1]][j], #the indexes of samples
                    [columns[i]]*len(j), #the size of the cluster
                    [i]*len(j), #the absolute number (ID) of the cluster
                    [len(j)]*len(j) #the number of samples in the cluster
                ]) for i,j in enumerate(selections)])
        cluster_subsets = out.T
        end = time.time()
        with open('./logging.txt', 'a+') as f:
            f.write(f'# of Refresh Samples: {cluster_subsets.shape}, step:{self.counter}, Select time: {end-start}\n')
        ##np.savez(f'./cluster_subsets_update',cluster_subsets)
        #output data as np for a subset of each cluster
        start = time.time()
        importance_collect = []
        importance_vars = []
        padding = self.batch_size - cluster_subsets.shape[0]%self.batch_size
        feed = np.concatenate([cluster_subsets, cluster_subsets[-padding-1:-1]])
        list_invar = {
            key: np.split(value, value.shape[0] // self.batch_size)
            for key, value in _DictDatasetMixin._idx_var(self.invar, feed[:,0]).items()
        }
        for i in range(len(next(iter(list_invar.values())))):
            importance = self.importance_measure(
                {key: value[i] for key, value in list_invar.items()}
            )
            importance_collect.append(importance[0])
            if not importance_vars:
                importance_vars = importance[1] ##the vars defined in importance function
        print(f"IMPORTANCE COLLECT FEED SHAPE: {feed.shape}")
        with open('./logging.txt', 'a+') as f:
            f.write(f'IMPORTANCE COLLECT FEED SHAPE: {feed.shape}')
        importance_collect = np.concatenate(importance_collect, axis= 0)[0:cluster_subsets.shape[0]]
        print(f"IMPORTANCE COLLECT SHAPE 2: {importance_collect.shape}")
        with open('./logging.txt', 'a+') as f:
            f.write(f", IMPORTANCE COLLECT SHAPE 2: {importance_collect.shape}, P.shape:{P.shape}\n")
        print(f"P.shape:{P.shape}")
        end = time.time()
        with open('./logging.txt', 'a+') as f:
            f.write(f'# of Refresh Samples: {cluster_subsets.shape}, step:{self.counter}, Calc time: {end-start}\n')
        return importance_collect, importance_vars, cluster_subsets
    
    def importance_update_invar(self,invar,P):
        #output data as np for a subset of each cluster
        importance_collect = []
        importance_vars = []
        padding = self.batch_size - P.shape[1]%self.batch_size
        epoch_range = np.concatenate([np.arange(P.shape[1]),#[np.random.permutation(P.shape[1]), #??????????? #BUG 
                np.random.choice(P.shape[1], padding, replace=False)])
        batch_range = np.split(epoch_range, len(epoch_range) // self.batch_size)
        print(f'batch_range:{epoch_range.shape[0]}, len: {len(batch_range)}, padding: {padding}')
        for i in batch_range:
            importance = self.importance_measure(
                {key: value[i] for key, value in invar.items()}
            )
            importance_collect.append(importance[0]) ##TODO figure this out better
            if not importance_vars:
                importance_vars = importance[1] ##the vars defined in importance function
        return np.concatenate(importance_collect, axis= 0)[0:P.shape[1]], importance_vars

    def dataset_update(self,importance_collect, importance_vars):
        try:
            output_dataset = _DictDatasetMixin._idx_var({
                k:v.cpu().detach().numpy() for k,v in self.invar.items()
                }, range(0,self.length))
        except:
            print("NO INPUT VARS BEING USED FOR OUTPUT DATASET")
            output_dataset = {}
        
        for k,i in enumerate(importance_vars):
            output_dataset[i] = importance_collect[:,k,None] #combine selected vars for output dataset as dict.
        
        self.TManager.update_dataset_outputs(output_dataset, self.counter)
        return

    def getBatchEpoch_level(self, P): 
        if not type(P) == csc_matrix:
            P = P.tocsc(copy=False)
        n = P.shape[1]
        print(f'BATCH EPOCH Clusters: {n}')
        out = np.concatenate([P.indices[P.indptr[i]:P.indptr[i+1]] for i in np.random.permutation(range(0,n))])
        print(f'BATCH EPOCH Shape {out.shape}')
        return out
    
    def getBatchClusterList_level(self, P): 
        if not type(P) == csc_matrix:
            P = P.tocsc(copy=False)
        n = P.shape[1]
        print(f'BATCH EPOCH Clusters: {n}')
        out = [P.indices[P.indptr[i]:P.indptr[i+1]] for i in np.random.permutation(range(0,n))]
        print(f'BATCH CLUSTER NUMBERS {len(out)}')
        print(f'BATCH CLUSTER NUMBERS {len(out)}')
        return out
    
    def centers_update(self):
        P = self.TManager.centroid_level_mapping
        if not type(P) == csc_matrix:
            P = P.tocsc(copy=False)
        #idxLists = [P.indices[P.indptr[i]:P.indptr[i+1]] for i in range(0,P.shape[1])]
        #print(f'idxLists')
        invar_center = None
        outvar_center = None
        lambda_weighting_center = None
        # invar_center = {k:
        #                     torch.stack([j[k].mean() for j in 
        #                     [_DictDatasetMixin._idx_var(self.invar, i) for i in idxLists]]
        #                     ).reshape(-1,1)
        #                 for k in self.invar.keys()}
        # print(f'invar_center')
        # outvar_center = {k:
        #                     torch.stack([j[k].mean() for j in 
        #                     [_DictDatasetMixin._idx_var(self.outvar, i) for i in idxLists]]
        #                     ).reshape(-1,1)
        #                 for k in self.outvar.keys()}
        # print(f'outvar_center')
        # lambda_weighting_center = {k:
        #                     torch.stack([j[k].mean() for j in 
        #                     [_DictDatasetMixin._idx_var(self.lambda_weighting, i) for i in idxLists]]
        #                     ).reshape(-1,1)
        #                 for k in self.lambda_weighting.keys()}
        print(f'lambda_weighting_center')

        cSizes=np.array([P.indptr[i+1]-P.indptr[i] for i in range(0,P.shape[1])])
        with open('./logging.txt', 'a+') as f:
            self.lend = time.time()
            f.write(f'Created Centroids, {self.lend-self.lstart}, step:{self.counter}  \n')
            #f.write(f"Centers: {invar_center['x'].shape} at K:{self.KNN} L:{self.coarse_level} \n")
            f.write(f"Centers: {len(cSizes)} at K:{self.KNN} L:{self.coarse_level} \n")
            f.write(f"Clusters Min: {np.min(cSizes)}, max:{np.max(cSizes)}, mean:{np.mean(cSizes)}, median: {np.median(cSizes)} \n")
            #print(f"Centers: {invar_center['x'].shape} at K:{self.KNN} L:{self.coarse_level}")
            print(f"Centers: {cSizes.sum()} at K:{self.KNN} L:{self.coarse_level}")
            self.lstart = time.time()
        
        self.TManager.update_centroid_vars(invar_center,outvar_center,lambda_weighting_center, P, cSizes, self.counter)

        #print(f"Initial centers made {invar_center['x'].shape}")
        print(f'LEVEL: {self.coarse_level}')

        #first_importance, first_importance_vars = self.importance_update_invar(invar_center,P) #first set of scores for spade input graph
        first_importance, first_importance_vars, first_cluster_subsets = self.importance_update_subsets(P, self.sample_ratio)
        print(f"Initial importance vars")
        self.TManager.update_centroid_values(first_importance,
                                              first_importance_vars, 
                                              first_cluster_subsets,
                                              self.counter) #save to TManager

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar: Dict[str, np.array],
        batch_size: int,
        importance_measure: Callable,
        lambda_weighting: Dict[str, np.array] = None,
        shuffle: bool = True,
        resample_freq: int = 1000,
        ################### New Args ##################
        warmup: int = 1000,
        initial_graph_vars: List[str] = [],
        graph_vars: List[str] = [],
        SPADE_vars: List[str] = [],
        LOSS_vars: List[str] = [],
        mapping_function: Union[Callable, str] = 'default',
        KNN: int = 10,
        sample_ratio: float = .2,
        sample_bounds: List[float] = [5/100,70/100],
        batch_iterations: int = 1000,
        cluster_size: int = 20,
        coarse_level: int = 1,
        iterations_rebuild: int = 20000,
        local_grid_width: int = 2,
        ###modified-add###
        batch_per_epoch: int = 1000,
    ):
        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

        print(f'local grid width dataset {local_grid_width}')
        self.batch_size = min(batch_size, self.length)
        self.shuffle = shuffle
        self.resample_freq = resample_freq
        self.importance_measure = importance_measure

        ## NEW
        self.warmup=warmup
        self.initial_graph_vars = initial_graph_vars
        print(f'Initial_Graph_Vars: {self.initial_graph_vars}')
        self.graph_vars = graph_vars
        print(f'Graph_Vars: {self.graph_vars}')
        self.SPADE_vars = SPADE_vars
        print(f'SPADE_vars: {self.SPADE_vars}')
        self.LOSS_vars = LOSS_vars
        print(f'LOSS_vars: {self.LOSS_vars}')
        self.mapping_function = mapping_function
        self.KNN = KNN
        self.sample_ratio = sample_ratio
        self.sample_bounds = sample_bounds

        self.batch_iterations = batch_iterations
        self.cluster_size = cluster_size

        self.coarse_level = coarse_level
        self.iterations_rebuild = iterations_rebuild
        self.batch_per_epoch = batch_per_epoch
        ##TODOj
        ## avg/median cluster size instead of coarse_level
        ## cluster_vars
        ## SPADE_vars
        ## removing distinction b/w initial and later
        ## epochs_resample (batch worker length)
        ## update to HyperEF
        
        self.local_grid_width = local_grid_width
        

        print('IMPORTS')
        
        self.checkInvar = self.invar.keys()
        self.TManager = MultiProcessSPADE(self.invar, self.batch_size, 
                            self.KNN, self.initial_graph_vars, self.graph_vars,
                            SPADE_vars, LOSS_vars,
                            grid_width = self.local_grid_width,
                            num_eigs = 2, SPADE_GRID = False, 
                            level = self.coarse_level,
                            r = 0, sample_ratio = self.sample_ratio,
                            sample_bounds = self.sample_bounds) 
        
        self.counter=0
        self.lstart=0
        self.lend=0
        
        ## TODOj do basics, get data, make graph, save backups, functions to check for them,
        
        def iterable_function():        

            ##Make sure output is actually different
            if all([i in self.invar.keys() for i in self.graph_vars]):
                #raise ValueError("At least 1 output var separate from input vars is needed.")
                print("NOTE: You are clustering subsequent graphs on INPUT VARS, these are static,\
                        this is wasting work until fully refreshing on subsets is implemented")

            #check inputs
            if not initial_graph_vars or not graph_vars:
                raise ValueError("Must have input and output vars selected")

            for i in self.warmup_loop(message='GRAPH NOT STARTED, WARMUP SAMPLING'):
                yield i
            print(f'WARMUP DONE')
            
            #importance_collect, importance_vars = self.importance_update()

            #self.dataset_update(importance_collect, importance_vars)
    
            ##self.lstart = time.time()
            print(__name__)
            self.TManager.new_clusters(detail = 'initial')

            #This loop will continue to feed randomized inputs until first SPADE calc is done.
            self.lstart = time.time()
            for i in self.warmup_loop(indicator_fcn = self.TManager.check_graph_clustering,
                                      threshold = True,
                                      detail='initial',
                                      message='FIRST GRAPH STARTED, WARMUP SAMPLING'):
                yield i
            print(f'FIRST GRAPH DONE')
            
            with open('./logging.txt', 'a+') as f:
                self.lend = time.time()
                f.write(f'First Graph After Warmup, {self.lend-self.lstart}, step:{self.counter}  \n')
                self.lstart = time.time()

            self.centers_update()

            # writer = SummaryWriter(log_dir='./')
            # gbatch = np.random.choice(self.length,self.batch_size)
            # invar = _DictDatasetMixin._idx_var(self.invar, gbatch) ##formerly [:,0]
            # outvar = _DictDatasetMixin._idx_var(self.outvar, gbatch)
            # lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, gbatch)
            # writer.add_graph(model, input_to_model=None, verbose=False, use_strict_trace=True)

            # for i in self.warmup_loop(message='Graph Done, iterating to next SPADE refresh', 
            #                           threshold=self.counter+(self.batch_iterations - self.counter%self.batch_iterations),
            #                           detail='For First Spade Calc'):
            #     yield i
            # print(f'STARTING SPADE')
            self.TManager.subset_refresh(#self.importance_update_invar, 
                        self.importance_update_subsets,
                        self.batch_size*self.batch_iterations,
                        self.counter)
            # self.lstart = time.time()
            # for i in self.warmup_loop(indicator_fcn = self.TManager.check_SPADE_progress,
            #                           threshold = True,
            #                           detail='initial',
            #                           message='FIRST SPADE CALC STARTED'):
            #     yield i
            # print(f'FIRST SPADE CALC DONE')
            
            # with open('./logging.txt', 'a+') as f:
            #     self.lend = time.time()
            #     f.write(f'First SPADE calc, {self.lend-self.lstart}, step:{self.counter}  \n')
            #     self.lstart = time.time()

            import os
            rebuild_reset = 0
            bstart = time.time()
            btime = []
            while True: #main loop
                with open('./batch_logs.txt', 'a+') as f:
                    bend = time.time()
                    f.write(f'Back in Outer Loop, {bend-bstart}, step:{self.counter}  \n')
                print(f'RUNNING? {self.TManager.running}, counter: {self.counter}')
                if (self.TManager.running == 0) and (self.counter%self.iterations_rebuild == 0 or rebuild_reset > 0):
                    #REBUILD_LARGE
                    #no graphs being made, iterations_rebuild met, full new clusters with output data.
                    print(f'ITERATIONS_REBUILD, OUTER LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                    importance_collect, importance_vars = self.importance_update()
                    try:
                        os.replace('./losses_all.npz', './losses_all_prev.npz')
                    except:
                        print('Failed to move losses_all')
                    np.savez('losses_all', importance_collect, importance_vars)
                    self.dataset_update(importance_collect, importance_vars)
                    print("Starting new Graph")
                    self.TManager.new_clusters()
                    rebuild_reset = 0
                elif self.TManager.running > 0: ##TODO make sure SPADE is ended
                    print("New graph not finished, resample current.")
                    rebuild_reset = 2
                while True: #inner loop, clusters and centroids are static until #REBUILD_LARGE is entered
                    if rebuild_reset == 1:
                        print(f'ITERATIONS_REBUILD, INNER LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                        break
                    elif rebuild_reset == 2:
                        print(f'GRAPH WAS NOT READY!{self.counter}, continuing')
                        rebuild_reset = 1
                    br_start = time.time()
                    with open('./batch_logs.txt', 'a+') as f:
                        bend = time.time()
                        f.write(f'Inner Loop, before batch_receiver, {bend-bstart}, step:{self.counter}  \n')
                    idx, backup_batch, prob = self.TManager.batch_receiver(self.batch_iterations) #new info from reciever
                    print(f'BATCH_RECEIVER: {idx}, {backup_batch}, {prob}')
                    if not backup_batch and prob is not None:
                        print(f'SHAPES {prob.shape},{self.invar["area"].shape}')
                        #pdb.set_trace()
                        assert len(prob) == len(self.invar["area"])    
                        prob = (prob / np.sum(self.invar["area"].numpy().reshape(-1) * prob))
                        print(f'PROB SHAPE: {prob.shape}')

                    br_end = time.time()
                    with open('./batch_logs.txt', 'a+') as f:
                        bend = time.time()
                        f.write(f'Inner Loop, after batch_receiver, {bend-bstart}, step:{self.counter}  \n')
                    if self.counter%500==0:
                        with open('./logging.txt', 'a+') as f:
                            f.write(f'batch_receiver, {br_end-br_start}, step:{self.counter}\n')
                    for i in range(0,idx.shape[0]//self.batch_size):
                        self.counter += 1
                        if self.counter%500 == 0:
                            print(f"running?{self.TManager.running}")
                        if self.counter%1000 == 0 and self.TManager.running:
                            with open('./batch_logs.txt', 'a+') as f:
                                bend = time.time()
                                f.write(f'About to check subprocesses, {bend-bstart}, step:{self.counter}  \n')
                            if self.TManager.check_graph_clustering(self.counter):
                                print("New graph done")
                                with open('./logging.txt', 'a+') as f:
                                    self.lend = time.time()
                                    f.write(f'New Graph, {self.lend-self.lstart}, step:{self.counter}\n')
                                    self.lstart = time.time()
                                    print('Loaded new graph')
                                self.centers_update()
                                break
                            with open('./batch_logs.txt', 'a+') as f:
                                bend = time.time()
                                f.write(f'About to check SPADE, {bend-bstart}, step:{self.counter}  \n')
                            if self.TManager.check_SPADE_progress(self.counter):
                                print("New SPADE done")
                                with open('./logging.txt', 'a+') as f:
                                    self.lend = time.time()
                                    f.write(f'SPADE CALC, {self.lend-self.lstart}, step:{self.counter}\n')
                                    self.lstart = time.time()
                                    print('Loaded new importance scores')
                                    #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                    #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                    #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                    #np.savez('losses_all',self.importance_update())
                                break
                            with open('./batch_logs.txt', 'a+') as f:
                                bend = time.time()
                                f.write(f'Checked Both, {bend-bstart}, step:{self.counter}  \n')
                        if (self.TManager.running == 0) and (self.counter%self.batch_iterations == 0):
                            print(f'SUBSET_REFRESH')
                            with open('./batch_logs.txt', 'a+') as f:
                                bend = time.time()
                                f.write(f'SUBSET_REFRESH Before:, {bend-bstart}, step:{self.counter}  \n')
                            self.TManager.subset_refresh(#self.importance_update_invar, 
                                                        self.importance_update_subsets,
                                                        self.batch_size*self.batch_iterations,
                                                        self.counter)
                            with open('./batch_logs.txt', 'a+') as f:
                                bend = time.time()
                                f.write(f'SUBSET_REFRESH After:, {bend-bstart}, step:{self.counter}  \n')
                            #if self.counter%10000 == 0:
                                #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                #print('#######REMOVE THIS BEFORE BENCHMARK RUNS##############')
                                #np.savez(f'losses_all_{self.counter}_subInterval',self.importance_update())
                        if self.counter%self.iterations_rebuild == 0:
                            print(f'ITERATIONS_REBUILD, INNER FOR LOOP, {self.counter}, setting: {self.iterations_rebuild}')
                            rebuild_reset = 1
                            break

                        # gather invar, outvar, and lambda weighting

                        batch_range = range(i*self.batch_size, i*self.batch_size + self.batch_size)
                        invar = _DictDatasetMixin._idx_var(self.invar, idx[batch_range,0]) ##formerly [:,0]
                        outvar = _DictDatasetMixin._idx_var(self.outvar, idx[batch_range,0])
                        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx[batch_range,0])

                        # set area value from importance sampling
                        #if (not self.TManager.running) and prob is not None:
                        if prob is not None:    
                            invar["area"] = (1.0 / (prob[idx[batch_range,0]] * self.batch_size))[:,None]
                            #pdb.set_trace()
                            with open('./area_logs.txt', 'a+') as f:
                                f.write(f'batch_area: {np.sum(invar["area"])}')
                        else:
                            invar["area"] *= self.batch_per_epoch  # random sampling
                            #pdb.set_trace()
                            with open('./area_logs.txt', 'a+') as f:
                                f.write(f'UNIFORM_BATCH_area: {np.sum(invar["area"].numpy())}')
                        #invar["area"] *= self.batch_per_epoch 
                        # return and count up
                        assert self.invar.keys() == self.checkInvar
                        bend = time.time()
                        btime.append((bend-bstart))
                        if (len(btime) > 1 and btime[-1] - btime[-2] > 1e-2):
                            with open('./batch_logs.txt', 'a+') as f:
                                f.write(f'----Average batch time: {np.average(btime[0:-1])}, total batch time: {np.sum(btime[0:-1])},'
                                        + f'step:{self.counter} to step:{self.counter-len(btime)}\n')
                                f.write(f'----Time since last batch:, {bend-bstart}, step:{self.counter}  \n')
                            btime = []
                        bstart = time.time()
                        yield (invar, outvar, lambda_weighting)
                    #if not backup_batch:
                        #break
                    #break

        self.iterable_function = iterable_function

        def save_dataset_importance_batch(invar):
            
            out = []
            list_importance = []
            list_invar = { ### variable: list of batches 
                key: np.split(value, value.shape[0] // self.batch_size)
                for key, value in invar.items()
            }
            for i in range(len(next(iter(list_invar.values())))):
                importance = self.importance_measure(
                    {key: value[i] for key, value in list_invar.items()}
                )
                list_importance.append(importance)
            cluster_values = np.concatenate(list_importance, axis= 0)

            out = cluster_values
            return out
            
        self.save_batch_function = save_dataset_importance_batch

    def __iter__(self):
        yield from self.iterable_function()


class ListIntegralDataset(_DictDatasetMixin, Dataset):
    """
    A map-style dataset for a finite set of integral training examples.
    """

    auto_collation = True

    def __init__(
        self,
        list_invar: List[Dict[str, np.array]],
        list_outvar: List[Dict[str, np.array]],
        list_lambda_weighting: List[Dict[str, np.array]] = None,
    ):
        if list_lambda_weighting is None:
            list_lambda_weighting = []
            for outvar in list_outvar:
                list_lambda_weighting.append(
                    {key: np.ones_like(x) for key, x in outvar.items()}
                )

        invar = _stack_list_numpy_dict(list_invar)
        outvar = _stack_list_numpy_dict(list_outvar)
        lambda_weighting = _stack_list_numpy_dict(list_lambda_weighting)

        super().__init__(invar=invar, outvar=outvar, lambda_weighting=lambda_weighting)

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        outvar = _DictDatasetMixin._idx_var(self.outvar, idx)
        lambda_weighting = _DictDatasetMixin._idx_var(self.lambda_weighting, idx)
        return (invar, outvar, lambda_weighting)

    def __len__(self):
        return self.length

    def save_dataset(self, filename):
        for idx in range(self.length):
            var_to_polyvtk(
                filename + "_" + str(idx).zfill(5),
                _DictDatasetMixin._idx_var(self.invar, idx),
            )


class ContinuousIntegralIterableDataset(IterableDataset):
    """
    An infinitely iterable dataset for a continuous set of integral training examples.
    This will resample training examples (create new ones) every iteration.
    """

    def __init__(
        self,
        invar_fn: Callable,
        outvar_fn: Callable,
        batch_size: int,
        lambda_weighting_fn: Callable = None,
        param_ranges_fn: Callable = None,
    ):

        self.invar_fn = invar_fn
        self.outvar_fn = outvar_fn
        self.lambda_weighting_fn = lambda_weighting_fn
        if lambda_weighting_fn is None:
            lambda_weighting_fn = lambda _, outvar: {
                key: np.ones_like(x) for key, x in outvar.items()
            }
        if param_ranges_fn is None:
            param_ranges_fn = lambda: {}  # Potentially unsafe?
        self.param_ranges_fn = param_ranges_fn

        self.batch_size = batch_size

        # TODO: re-write iterable function so that for loop not needed - to improve performance

        def iterable_function():
            while True:
                list_invar = []
                list_outvar = []
                list_lambda_weighting = []
                for _ in range(self.batch_size):
                    param_range = self.param_ranges_fn()
                    list_invar.append(self.invar_fn(param_range))
                    if (
                        not param_range
                    ):  # TODO this can be removed after a np_lambdify rewrite
                        param_range = {"_": next(iter(list_invar[-1].values()))[0:1]}

                    list_outvar.append(self.outvar_fn(param_range))
                    list_lambda_weighting.append(
                        self.lambda_weighting_fn(param_range, list_outvar[-1])
                    )
                invar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_invar))
                outvar = Dataset._to_tensor_dict(_stack_list_numpy_dict(list_outvar))
                lambda_weighting = Dataset._to_tensor_dict(
                    _stack_list_numpy_dict(list_lambda_weighting)
                )
                yield (invar, outvar, lambda_weighting)

        self.iterable_function = iterable_function

    def __iter__(self):
        yield from self.iterable_function()

    @property
    def invar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        return list(invar.keys())

    @property
    def outvar_keys(self):
        param_range = self.param_ranges_fn()
        invar = self.invar_fn(param_range)
        outvar = self.outvar_fn(invar)
        return list(outvar.keys())

    def save_dataset(self, filename):
        # Cannot save continuous data-set
        pass


class DictVariationalDataset(Dataset):
    """
    A map-style dataset for a finite set of variational training examples.
    """

    auto_collation = True

    def __init__(
        self,
        invar: Dict[str, np.array],
        outvar_names: List[str],  # Just names of output vars
    ):

        self.invar = Dataset._to_tensor_dict(invar)
        self.outvar_names = outvar_names
        self.length = len(next(iter(invar.values())))

    def __getitem__(self, idx):
        invar = _DictDatasetMixin._idx_var(self.invar, idx)
        return invar

    def __len__(self):
        return self.length

    @property
    def invar_keys(self):
        return list(self.invar.keys())

    @property
    def outvar_keys(self):
        return list(self.outvar_names)

    def save_dataset(self, filename):
        for i, invar in self.invar.items():
            var_to_polyvtk(invar, filename + "_" + str(i))


def _stack_list_numpy_dict(list_var):
    var = {}
    for key in list_var[0].keys():
        var[key] = np.stack([v[key] for v in list_var], axis=0)
    return var

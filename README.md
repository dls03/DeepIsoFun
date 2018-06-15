# DeepIsoFun

DeepIsoFun: A deep domain adaptation approach to predict isoform functions

Isoforms are mRNAs produced from the same gene locus by alternative splicing and may have different functions. Although gene functions have been studied extensively, little is known about the specific functions of isoforms. Recently, some computational approaches based on multiple instance learning have been proposed to predict isoform functions from annotated gene functions and expression data, but their performance is far from being desirable primarily due to the lack of labeled training data. To improve the performance on this problem, we propose a novel deep learning method, DeepIsoFun, that combines multiple instance learning with domain adaptation. The latter technique helps to transfer the knowledge of gene functions to the prediction of isoform functions and provides additional labeled training data. Our model is trained on a deep neural network architecture so that it can adapt to different expression distributions associated with different gene ontology terms.


### Install Caffe 

##### Download
Download the master branch of [Caffe](http://caffe.berkeleyvision.org/) and compile it in your machine. See here for the [installation](http://caffe.berkeleyvision.org/installation.html) guide. </br>  
For some systems current version of caffe might not work. You can try older [release](https://github.com/BVLC/caffe/releases).

##### Makefile Adjustment
In Makefile.config file:</br>
- `USE_CUDNN := 1` (uncomment to build with cuDNN) </br>
- `CPU_ONLY := 1` (uncomment to build without GPU support) </br>
- `WITH_PYTHON_LAYER := 1` (uncomment to support layers written in Python) </br>
- `PYTHON_INCLUDE := /usr/include/python2.7` (Adjust Python path. We need to be able to find Python.h and numpy/arrayobject.h.) </br>
- `CUDA_DIR := /opt/linux/centos/7.x/x86_64/pkgs/cuda/7.0/` (Adjust CUDA directory. CUDA directory contains bin/ and lib/ directories that we need.) </br>
- Adjust BLAS path for include and lib directories </br>
```
BLAS_INCLUDE := /caffe/openblas/include
BLAS_LIB := /caffe/openblas/lib
```

Example: DeepIsoFun [Makefile.config]()

##### Include Layers
- [Gradient revarsal layer cpp](https://github.com/ddtm/caffe/blob/grl/src/caffe/layers/gradient_scaler_layer.cpp) and 
[Gradient revarsal layer cpp](https://github.com/ddtm/caffe/blob/grl/src/caffe/layers/gradient_scaler_layer.cu).
 We have used this layer from this paper (Ganin et al. 2015)(http://proceedings.mlr.press/v37/ganin15.pdf) </br>
- [Multiple instance loss layer] () </br>
- [Multiple instance loss layer] () </br>
- [Multiple instance loss layer] () </br>
Put these files into /caffe-master/src/caffe/layers/ </br>
- [gradient_scaler_layer.hpp] () : Put these files into /caffe-master/include/caffe/layers/</br>
- [layer.hpp] () : Put these files into /caffe-master/include/caffe/</br>
- [messenger.hpp] () : Put these files into /caffe-master/include/caffe/ </br>

##### Prerquired tools
- [CUDA](https://developer.nvidia.com/cuda-zone) </br> 
- [OpenBLAS](http://www.openblas.net/) </br> 
- [OpenCV](https://opencv.org/) </br> 
- [Boost](https://www.boost.org/) </br> 
- [Python and pycaffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html) </br>  

To Compile and test run these commands:
```
$ make all
$ make test 
$ make runtest
```
<\br>
Tips: Make sure you have compiled successfully everything mentioned above. 

### Data
- Expression profile of isoforms and genes.</br>
	- [Isoform Expression Data] ()
	- [Gene Expression Data] ()
- ID Conversion 
	- [Gene Isoform relations] () 
- Get GO annotation for gene. </br>
	- [GO Annotation] ()
- Get Gene Ontology Hierarchy file to get parent-child relationship of GO terms. </br>
	- [GO Hierarchy] ()
- GO set </br>
	- [All GO] ()
	- [GO slim] ()
- SRA files 
	- [Final set of SRA files] ()
- Data Preprocessing
	- [] ()
	- [] ()
	
### Run DeepIsoFun
Run the script file [./runM.sh] ()





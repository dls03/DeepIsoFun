# DeepIsoFun

DeepIsoFun: A deep domain adaptation approach to predict isoform functions

Isoforms are mRNAs produced from the same gene locus by alternative splicing and may have different functions. Although gene functions have been studied extensively, little is known about the specific functions of isoforms. Recently, some computational approaches based on multiple instance learning have been proposed to predict isoform functions from annotated gene functions and expression data, but their performance is far from being desirable primarily due to the lack of labeled training data. To improve the performance on this problem, we propose a novel deep learning method, DeepIsoFun, that combines multiple instance learning with domain adaptation. The latter technique helps to transfer the knowledge of gene functions to the prediction of isoform functions and provides additional labeled training data. Our model is trained on a deep neural network architecture so that it can adapt to different expression distributions associated with different gene ontology terms.


### Installation

#### Download
Download the master branch of [Caffe](http://caffe.berkeleyvision.org/) and compile it on your machine. See here for the [installation](http://caffe.berkeleyvision.org/installation.html) guide. </br>  
For some systems, the current version of Caffe might not work. You can try an older [release](https://github.com/BVLC/caffe/releases).

#### Makefile Adjustment
In the Makefile.config file:</br>
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

#### Included Layers
- [Gradient revarsal layer cpp](https://github.com/ddtm/caffe/blob/grl/src/caffe/layers/gradient_scaler_layer.cpp) and 
[Gradient revarsal layer cpp](https://github.com/ddtm/caffe/blob/grl/src/caffe/layers/gradient_scaler_layer.cu).
 This layer is from the paper (Ganin et al. 2015)(http://proceedings.mlr.press/v37/ganin15.pdf) </br>
- [Multiple instance loss layer] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/MIloss.py) </br>
- [LSE loss layer] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/LSEloss.py) </br>
- [GM loss layer] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/GMloss.py) </br>
Put these files in /caffe-master/src/caffe/layers/ </br>
- [gradient_scaler_layer.hpp] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/gradient_scaler_layer.hpp) : Put these files in /caffe-master/include/caffe/layers/</br>
- [layer.hpp] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/layer.hpp) : Put these files in /caffe-master/include/caffe/</br>
- [messenger.hpp] (https://github.com/dls03/DeepIsoFun/blob/master/Layers/messenger.hpp) : Put these files in /caffe-master/include/caffe/ </br>

#### Required Supporting Systems
- [CUDA](https://developer.nvidia.com/cuda-zone) </br> 
- [OpenBLAS](http://www.openblas.net/) </br> 
- [OpenCV](https://opencv.org/) </br> 
- [Boost](https://www.boost.org/) </br> 
- [Python and pycaffe](http://caffe.berkeleyvision.org/tutorial/interfaces.html) </br>  

To Compile and test DeepIsoFun, run the following commands:
```
$ make all
$ make test 
$ make runtest
```

*Tips: Make sure you have included all the layers and compiled Caffe successfully.*  

### Data
- ID conversion 
	- [Gene-isoform relationship data] (https://github.com/dls03/DeepIsoFun/blob/master/Data/GeneIsoformNameNew) 
- GO annotation for genes </br>
	- [GO annotation] (https://github.com/dls03/DeepIsoFun/blob/master/Data/GeneAnnoNewH)
- Parent-child relationship among GO terms </br>
	- [GO hierarchy] (http://www.geneontology.org/ontology/go-basic.obo)
- GO set </br>
	- [All GO] (https://github.com/dls03/DeepIsoFun/blob/master/Data/go_6up)
	- [GO slim] (https://github.com/dls03/DeepIsoFun/blob/master/Data/goslim.txt)
- SRA files 
	- [Final set of SRA files] (https://github.com/dls03/DeepIsoFun/blob/master/Data/sra_filter_final.txt)
- Data preprocessing
	- [Download SRA read file] (https://github.com/dls03/DeepIsoFun/blob/master/Preprocessing%20Tools/dw_SRAdata.R)
	- [Run Kallisto to generate expression profile] (https://github.com/dls03/DeepIsoFun/blob/master/Preprocessing%20Tools/kalrun.R)
	
### Run DeepIsoFun
Run the script file `./runM.sh` (https://github.com/dls03/DeepIsoFun/tree/master/DeepIsoFun) </br>
It will generate the prediction for all isoforms, the AUC and AUPRC values for each GO term. (https://github.com/dls03/DeepIsoFun/blob/master/Results/go_auc_auprc_deepisofun.txt) 





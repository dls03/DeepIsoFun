import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['GLOG_minloglevel'] = '1'
import caffe
from caffe import layers as L, params as P # Shortcuts to define the net prototxt.
sys.path.append("DeepIsoFun/") # the datalayers we will use are in this directory.
sys.path.append("pycaffe") # the tools file is in this folder
import h5py
import shutil
import tempfile
import sklearn
import sklearn.datasets
import sklearn.linear_model
import sklearn.cross_validation
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn import preprocessing
import pandas as pd
import math 
import copy
import random
import misvm
import time
from sklearn.manifold import TSNE
#from ggplot import *
from sklearn.model_selection import KFold
from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
#from matplotlib import pyplot as plt
#from tsne import bh_sne


######################################### Set path for Input and Output data #########################################################

go_list='./Data/goslim.txt'
go_names=pd.read_csv(go_list, sep='\t')
go_names=np.asarray(go_names)

go_list_done='./Data/goslim_done.txt'
go_names_done=pd.read_csv(go_list_done, sep='\t')
go_names_done=np.asarray(go_names_done)

print(go_names_done);

#sig=readLines('./Data/sig.txt')
#mig=readLines('./Data/mig.txt')

predchild_file='./new_data/predchild_deepisofun.txt'
child_pred='./new_data/childpred_deepisofun.txt'
auc_fileout='./new_data/aucmic_deepisofun.txt'
prauc_fileout='./new_data/praucmic_deepisofun.txt'

sig=np.loadtxt('./new_data/sig.txt', dtype=int, delimiter="\t")
mig=np.loadtxt('./new_data/mig.txt', dtype=int, delimiter="\t")

gene_file='./Data/geneExprNewNorm'
iso_file='./Data/isoExprNewNorm'

gene_set=np.loadtxt(gene_file, dtype=float, delimiter="\t", skiprows=1)
iso_set=np.loadtxt(iso_file, dtype=float, delimiter="\t", skiprows=1)

#fout_auc=open(auc_fileout,'w')
#fout_auc=open(auc_fileout,'w')
#fout_auc.close()
fout_prauc=open(prauc_fileout,'a')
fout_predchild=open(predchild_file,'a')
fout_child_pred=open(child_pred,'a')

#fout_auc=open(auc_fileout,'a')
#fout_auc.write(it + '\t')
#fout_auc.close()

################################################################################################################################################



###################################################### Read GO hierarchy helper function #########################################################
## Read the .obo file 
def getTerm(stream):
  block = []
  for line in stream:
    if line.strip() == "[Term]" or line.strip() == "[Typedef]":
      break
    else:
      if line.strip() != "":
        block.append(line.strip())
  return block

## Split and parse each go term
def parseTagValue(term):
  data = {}
  for line in term:
    tag = line.split(': ',1)[0]
    value = line.split(': ',1)[1]
    if not data.has_key(tag):
      data[tag] = []
    data[tag].append(value)
  return data

## Get all descendents of a go term  
def getDescendents(goid):
  recursiveArray = [goid]
  if terms.has_key(goid):
    children = terms[goid]['c']
    if len(children) > 0:
      for child in children:
        recursiveArray.extend(getDescendents(child))
  return set(recursiveArray)

## Get all ancestor of a go term
def getAncestors(goid):
  recursiveArray = [goid]
  if terms.has_key(goid):
    parents = terms[goid]['p']
    if len(parents) > 0:
      for parent in parents:
        recursiveArray.extend(getAncestors(parent))
  return set(recursiveArray)

#    
#pcrelation=copy.deepcopy(terms)
#
#########################################################################################################################################



#########################################################################################################################################
## Adjust range to get result on different neuron size. 600-1000 is optimal neuron size
for neuronL1 in range(600,1000,200):
    
#    fout_auc=open(auc_fileout,'a')
#    fout_auc.write( str(neuronL1) + '\n' + '\n' + '\n')
#    fout_auc.close()
    
    oboFile = open('go-basic.obo','r')

    #declare a blank dictionary
    #keys are the goids
    terms = {}

    #skip the file header lines
    getTerm(oboFile)

    #infinite loop to go through the obo file.
    #Breaks when the term returned is empty, indicating end of file
    while 1:
      term = parseTagValue(getTerm(oboFile))
      if len(term) != 0:
        termID = term['id'][0]
        if term.has_key('is_a'):
          termParents = [p.split()[0] for p in term['is_a']]
          if not terms.has_key(termID):
            #each goid will have two arrays of parents and children
            terms[termID] = {'p':[],'c':[]}
          #append parents of the current term
          terms[termID]['p'] = termParents
          #for every parent term, add this current term as children
          for termParent in termParents:
            if not terms.has_key(termParent):
              terms[termParent] = {'p':[],'c':[]}
            terms[termParent]['c'].append(termID)
      else:
        break
		
    pcrelation=copy.deepcopy(terms)
    
    
    
    ## Each j represent depth of GO relationship tree. j=20 is enough to cover whole tree.
    for j in range(0,20):
        child=[]
        for key,value in terms.items():
            if(len(value.values()[1])==0):
                #print key,value
                child.append(key)
        print len(child)
        #print len(terms)
        for chi in child:
            parent=terms[chi]['p']
            #print parent
            for p in parent:
                terms[p].values()[1].remove(chi)
        for chi in child:
            del terms[chi]

        df=pd.DataFrame()
        df_pred=pd.DataFrame()
##########################################################################################################################################

		## Each it represent a GO term
        for it in child: #label_gene_mic9254   #868mf, 2887bp, 469cc
			
			############################################### preprocessing data and initialize variables ###################################
			###############################################################################################################################
            if it not in go_names:
                continue
            if it in go_names_done:
                continue
            fout_auc=open(auc_fileout,'a')
            fout_auc.write(it + '\t')
            fout_auc.close()
            GOID=it.replace(':','.');
            print(it)
            print(GOID)
            IsoAnno=pd.read_csv('./Data/IsoAnnoNewH', sep="\t")
            a=(IsoAnno.GO_ID==it)*1
            IsoAnno.GO_ID=a;
            
			gene_lab = IsoAnno.groupby(['geneID'])['GO_ID'].sum()
            iso_lab = IsoAnno.groupby(['isoformID','geneID'])['GO_ID'].sum()
			
            gene_lab[gene_lab>0]=1
            iso_lab[iso_lab>0]=1

            example_gene=[(index , val) for index, val in gene_lab.iteritems()]
            example_set=[(index[1], index[0] ,val) for index, val in iso_lab.iteritems()]

            example_gene=np.asarray(example_gene)
            example_set=np.asarray(example_set)

            term_size_temp=0
            term_size_temp=np.sum(example_gene[:,1])

            print term_size_temp
            if (term_size_temp<5 or term_size_temp>3000):
                print term_size_temp
                continue

            ##Hierarchy Child Prediction Merge
            GOID_t=GOID.replace('.',':');
            child_GOID=pcrelation[it]['c']

            with open(predchild_file, 'r') as f:
                        go_child_line = f.readline()

            all_child_go_name=go_child_line.split("\t")

            for c_i in child_GOID:
                print(c_i)
                c_i=c_i.replace(':','.');
                if c_i in all_child_go_name:
                    go_child_index=all_child_go_name.index(c_i)
                    print(go_child_index)
                    iso_child_label=np.loadtxt(predchild_file, dtype=float, delimiter="\t", skiprows=1, usecols = ([go_child_index]))
                    example_set[:,2]=np.logical_or(example_set[:,2],iso_child_label)
                else:
                    print('child yet not predicted\n')


            #example_set=np.loadtxt(file, dtype=float, delimiter="\t")

            auc_all=[]
            midann_auc_all=[]
            midann_pr_auc_all=[]
            midann_auc_all_mig=[]
            midann_pr_auc_all_mig=[]
			GO_term_size=0;
            final_prediction = pd.DataFrame(
                        {'isoformID': [0],
                        'truelabel': [0],
                        'predlabel': [0]
                        })
            final_prediction_pred = pd.DataFrame(
                        {'isoformID': [0],
                        'truelabel': [0],
                        'predlabel': [0]
                        }) 
            example_set_copy=copy.deepcopy(example_set)
            #example_set_copy=example_set_copy.astype(float)
			
            start_time=time.time() #### Calculate the time needed
#            for itercv in range(5):
            for itera in range(1736):    
		
                auc_all=[]
    	        midann_auc_all=[]
                midann_pr_auc_all=[]
            	midann_auc_all_mig=[]
                midann_pr_auc_all_mig=[]
            	GO_term_size=0;
				example_set=example_set_copy;
				gene_c=19532 #gene size
                iso_c=47393 #isoform size
                source_bags=gene_set[:,1+itera] # gene expression
                target_bags_o=iso_set[:,2+itera] # isoform expression
                s_labels = example_gene[:,1]   #gebe label
                t_labels = example_set[:,2] # isoform label
                baglabels = example_set[:,0] #gene bag label
                instancelabels = example_set[:,1] #isoform bag label

                s_labels[s_labels<1]=0
                t_labels[t_labels<1]=0

                GO_term_size=np.sum(s_labels)
                po_gene=np.sum(s_labels)
                po_iso=np.sum(t_labels)

                source_bags=pd.DataFrame(source_bags)
                target_bags_o=pd.DataFrame(target_bags_o)
                s_labels=pd.DataFrame(s_labels)
                t_labels=pd.DataFrame(t_labels)
                baglabels=pd.DataFrame(baglabels)
                instancelabels=pd.DataFrame(instancelabels)

                #print po_gene;
                #print po_iso;
                #print 'data augmentation start'
                rep_g=gene_c/po_gene;
                rep_i=iso_c/po_iso;
                po_index=(s_labels[0]>0)
                po_index_t=(t_labels[0]>0)

                #print s_labels
                #print t_labels
                #print po_index
                #print po_index_t
                #print s_labels[po_index]
                #print t_labels[po_index_t]
                source_data=np.asarray(source_bags.append([source_bags[po_index]]*rep_g, ignore_index=True))
                target_bags=np.asarray(target_bags_o.append([target_bags_o[po_index_t]]*rep_i, ignore_index=True))
                source_labels=np.asarray(s_labels.append([s_labels[po_index]]*rep_g, ignore_index=True))
                targetlabels=np.asarray(t_labels.append([t_labels[po_index_t]]*rep_i, ignore_index=True))
                targetbaglabels=np.asarray(baglabels.append([baglabels[po_index_t]]*rep_i, ignore_index=True))
                targetinstancelabels=np.asarray(instancelabels.append([instancelabels[po_index_t]]*rep_i, ignore_index=True))

                print(source_data.shape)
                print(source_labels.shape)
                ##print(source_bag_labels.shape)
                print(target_bags.shape)
                print(targetlabels[targetlabels>0])

				###Discard
                #source_data = preprocessing.scale(source_data);
                #target_bags = preprocessing.scale(target_bags);

                #target_data=target_bags;
                #test_data=target_bags;
                #target_labels=target_labels;
                #test_labels=target_labels;
                #target_bag_labels=target_bag_labels;
                #test_bag_labels=target_bag_labels;
				############################################################################################################################
				############################################################################################################################
				
				
				
				########################################## 5-fold cross validation #########################################################
                kf = KFold(n_splits=5)
                kf.get_n_splits(target_bags)
                #print(kf)
                #KFold(n_splits=2, random_state=None, shuffle=False)
                for train_index, test_index in kf.split(target_bags):

                    print(source_data.shape)
                    print(source_labels.shape)

                    print(target_bags.shape)
                    print(targetlabels.shape)

                    print("TRAIN:", train_index, "TEST:", test_index)
                    #X_train, X_test = X[train_index], X[test_index]
                    #y_train, y_test = y[train_index], y[test_index]
                    target_data, test_data = target_bags[train_index], target_bags[test_index]
                    target_labels, test_labels = targetlabels[train_index], targetlabels[test_index]
                    target_bag_labels, test_bag_labels = targetbaglabels[train_index], targetbaglabels[test_index]
                    target_instance_labels, test_instance_labels = targetinstancelabels[train_index], targetinstancelabels[test_index]

                    #target_data, test_data, target_labels, test_labels, target_bag_labels, test_
                    #sklearn.cross_validation.train_test_split(target_bags,target_labels, target_\
                    #target_data, test_data, target_labels, test_labels, target_bag_labels, test_bag_labels, target_instance_labels, test_instance_labels = sklearn.cross_validation.train_test_split(target_bags,target_labels, target_bag_labels, target_instance_labels)


                    print('Size of source, target and test')
                    print(source_data.shape)
                    print(target_data.shape)
                    print(test_data.shape)

                    lvsize=1;
					########################################################################################################################

					
                    ####################### Write out the data to HDF5 files in a temp directory. ##########################################
                    dirname = os.path.abspath('./examples/midann/data/')
                    if not os.path.exists(dirname):
                        os.makedirs(dirname)

                    source_filename = os.path.join(dirname, 'source.h5')
                    target_filename = os.path.join(dirname, 'target.h5')
                    test_filename = os.path.join(dirname, 'test.h5')
					########################################################################################################################
					
					
					
                    ####################### HDF5DataLayer source should be a file containing a list of HDF5 filenames.
                    ####################### To show this off, we'll list the same data file twice.
                    with h5py.File(source_filename, 'w') as f:
                        f.create_dataset('source_data', (source_data.shape[0],source_data.shape[1]),  data=source_data)
                        f.create_dataset('lp_label', (source_data.shape[0],lvsize),  data=source_labels)
                        #f.create_dataset('bag_label', (source_data.shape[0],lvsize),  data=source_bag_labels.astype(np.int_))
                    with open(os.path.join(dirname, 'source.txt'), 'w') as f:
                        f.write(source_filename + '\n')
					############################################################################################################################
                    
					
					
                    ####################### HDF5 is pretty efficient, but can be further compressed.
                    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
                    with h5py.File(target_filename, 'w') as f:
                        f.create_dataset('target_data',(target_data.shape[0],target_data.shape[1]), data=target_data)
                        f.create_dataset('lp_target_label',(target_data.shape[0],lvsize), data=target_labels)
                        f.create_dataset('bag_target_label',(target_data.shape[0],lvsize), data=target_bag_labels.astype(np.int_))
                        f.create_dataset('instance_target_label',(target_data.shape[0],lvsize), data=target_instance_labels.astype(np.int_))
                    with open(os.path.join(dirname, 'target.txt'), 'w') as f:
                        f.write(target_filename + '\n')

                    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
                    with h5py.File(test_filename, 'w') as f:
                        f.create_dataset('data',(test_data.shape[0],test_data.shape[1]), data=test_data)
                        f.create_dataset('lp_label',(test_data.shape[0],lvsize), data=test_labels)
                        f.create_dataset('bag_label',(test_data.shape[0],lvsize), data=test_bag_labels.astype(np.int_))
                        f.create_dataset('instance_label',(test_data.shape[0],lvsize), data=test_instance_labels.astype(np.int_))
                    with open(os.path.join(dirname, 'test.txt'), 'w') as f:
                        f.write(test_filename + '\n')
					###################################################################################################################
					
					
					
					###########################################Discard###################################################################
                    #X, Xt, S, St, y, yt = sklearn.cross_validation.train_test_split(X,S,y)
                    ########################### Train and test the scikit-learn SGD logistic regression.
                    #clfe = sklearn.linear_model.SGDClassifier(
                    #    loss='log', n_iter=1000, penalty='l2', alpha=5e-4 , class_weight='auto')
					####################################################################################################################
					
					
					
					############################################## Model Train #########################################################
                    def modeltrain(hdf5s,hdf5t, batch_size):
                        #logistic regression: data, matrix multiplication, and 2-class softmax loss
                        n = caffe.NetSpec()
                        n.source_data, n.lp_label= L.HDF5Data(batch_size=batch_size, source=hdf5s, ntop=2, shuffle=False)
                        n.source_domain_labels= L.DummyData(data_filler=dict(type='constant', value=0), num=batch_size, channels=1, height=1, width=1)
                        #n.target_data, n.lp_target_label, n.bag_target_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=3, shuffle=False)
                        #n.target_data, n.lp_target_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=2, shuffle=False)
                        #n.target_data, n.lp_target_label, n.instance_target_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=3, shuffle=False)
                        n.target_data, n.lp_target_label, n.bag_target_label, n.instance_target_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=4, shuffle=False)
                        n.target_domain_labels=L.DummyData(data_filler=dict(type='constant', value=1), num=batch_size, channels=1, height=1, width=1)
                        bottom_layers_data=[n.source_data, n.target_data]
                        n.data=L.Concat(*bottom_layers_data, concat_dim=0)
                        bottom_layers_domain=[n.source_domain_labels, n.target_domain_labels]
                        n.dc_label=L.Concat(*bottom_layers_domain, concat_dim=0)
                        n.ip1= L.InnerProduct(n.data, num_output=neuronL1, weight_filler=dict(type='xavier'))
                        n.relu1 = L.Sigmoid(n.ip1, in_place=True)
                        #n.dropout1 = L.Dropout(n.relu1, dropout_ratio=0.5)
                        n.ip2= L.InnerProduct(n.relu1, num_output=neuronL1-400, weight_filler=dict(type='xavier'))
                        n.source_feature, n.target_feature = L.Slice(n.ip2, slice_dim=0, ntop=2)
                        #L.Silence(n.target_feature);
                        #clfe.fit(n.source_feature, n.lp_label)	
                        #n.real, n.ip3 = L.Python(n.target_feature, n.lp_target_label, n.bag_label, module= 'missSVM', layer='missSVMLayer', ntop=2)
                        n.ip3 = L.InnerProduct(n.source_feature, num_output=1, weight_filler=dict(type='xavier'))
                        #n.ip3=L.Sigmoid(n.ip33, in_place=True)
                        n.ip4= L.InnerProduct(n.target_feature, num_output=1, weight_filler=dict(type='xavier'))
                        #n.ip5=L.Sigmoid(n.ip4, in_place=True)
                        #n.ll=clfe.predict(n.source_feature)
                        #n.accuracy = L.Accuracy(n.ip4, n.lp_target_label)
                        #n.losslp = L.Python(n.ip4, n.lp_target_label, n.bag_target_label, module = 'GMloss', layer='MultipleInstanceLossLayer')
                        #n.P, n.Y = L.Python(n.ip4, n.lp_target_label, n.bag_target_label, module = 'MIloss', layer='MultipleInstanceLossLayer', ntop=2) 
                        #n.losslp = L.SigmoidCrossEntropyLoss(n.P, n.Y)
                        n.losslp = L.SigmoidCrossEntropyLoss(n.ip4, n.lp_target_label)
                        n.losslps = L.SigmoidCrossEntropyLoss(n.ip3, n.lp_label)
                        n.grl= L.GradientScaler(n.ip2, lower_bound=0.0)
                        n.ip11= L.InnerProduct(n.grl, num_output=300, weight_filler=dict(type='xavier'))
                        n.relu11 = L.Sigmoid(n.ip11, in_place=True)
                        n.dropout11 = L.Dropout(n.relu11, dropout_ratio=0.5)
                        n.ip12 = L.InnerProduct(n.dropout11, num_output=1, weight_filler=dict(type='xavier'))
                        #n.final = L.Sigmoid(n.ip12, in_place=True)
                        n.lossdc = L.SigmoidCrossEntropyLoss(n.ip12, n.dc_label, loss_weight=0.1)
                        return n.to_proto()
					################################################################################################################################

					

					####################################################### Model Test ##############################################################
                    def modeltest(hdf5s, hdf5t, batch_size):
                        #logistic regression: data, matrix multiplication, and 2-class softmax loss
                        n = caffe.NetSpec()
                        #n.data, n.lp_label, n.bag_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=3)
                        #n.data, n.lp_label, n.instance_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=3)
                        #n.data, n.lp_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=2)
                        n.data, n.lp_label, n.bag_label, n.instance_label = L.HDF5Data(batch_size=batch_size, source=hdf5t, ntop=4)
                        n.dc_label=L.DummyData(data_filler=dict(type='constant', value=1), num=batch_size, channels=1, height=1, width=1)
                        n.ip1 = L.InnerProduct(n.data, num_output=neuronL1, weight_filler=dict(type='xavier'))
                        n.relu1 = L.Sigmoid(n.ip1, in_place=True)
                        #n.dropout1 = L.Dropout(n.relu1, dropout_ratio=0.5)
                        n.ip2 = L.InnerProduct(n.relu1, num_output=neuronL1-400, weight_filler=dict(type='xavier'))
                        n.target_feature=L.Split(n.ip2)
                        n.ip4 = L.InnerProduct(n.target_feature, num_output=1, weight_filler=dict(type='xavier'))
                        #n.ip5=L.Sigmoid(n.ip4, in_place=True)
                        #n.real, n.ip3 = L.Python(n.source_feature, n.lp_label, n.bag_label, module= 'missSVM', layer='missSVMLayer', ntop=2)
                        #n.ip3 = L.InnerProduct(n.source_feature, num_output=1, weight_filler=dict(type='xavier'))
                        #n.accuracy = L.Accuracy(n.ip4, n.lp_label)
                        #L.Silence(n.bag_label);
                        #n.losslp = L.Python(n.ip4, n.lp_label, n.bag_label, module = 'GMloss', layer='MultipleInstanceLossLayer')
                        #n.P , n.Y = L.Python(n.ip4, n.lp_label, n.bag_label, module = 'MIloss', layer='MultipleInstanceLossLayer', ntop=2)
                        #n.losslp = L.SigmoidCrossEntropyLoss(n.P, n.Y)
                        n.losslp = L.SigmoidCrossEntropyLoss(n.ip4, n.lp_label)
                        #n.losstlp = L.SigmoidCrossEntropyLoss(n.ip4, n.lp_label)
                        n.grl= L.GradientScaler(n.ip2, lower_bound=0.0)
                        n.ip11 = L.InnerProduct(n.grl, num_output=300, weight_filler=dict(type='xavier'))
                        n.relu11 = L.Sigmoid(n.ip11, in_place=True)
                        n.dropout11 = L.Dropout(n.relu11, dropout_ratio=0.5)
                        n.ip12 = L.InnerProduct(n.dropout11, num_output=1, weight_filler=dict(type='xavier'))
                        n.lossdc = L.SigmoidCrossEntropyLoss(n.ip12, n.dc_label, loss_weight=0.1) 
                        return n.to_proto()
					#######################################################################################################################################

					

					##################################################### train path ######################################################################
                    train_net_path = 'Configuration/py_train.prototxt'
                    with open(train_net_path, 'w') as f:
                        f.write(str(modeltrain('Configuration/source.txt', 'examples/midann/data/target.txt', 200)))# source_data.shape[0])))
					#######################################################################################################################################
					
					
					
					##################################################### test path #######################################################################
                    test_net_path = 'Configuration/py_test.prototxt'
                    with open(test_net_path, 'w') as f:
                        f.write(str(modeltest('Configuration/test.txt','examples/midann/data/test.txt', 200)))#test_data.shape[0])))
					#######################################################################################################################################
					
					

                    ##################################################### define solver ###################################################################
                    from caffe.proto import caffe_pb2

                    def solver(train_net_path , test_net_path):
                        s=caffe_pb2.SolverParameter()
                        s.train_net = train_net_path
                        s.test_net.append(test_net_path)
                        s.test_interval = 3 
                        s.test_iter.append(15)
                        s.max_iter = 100 
                        s.base_lr =0.001
                        s.lr_policy='step'
                        s.gamma=0.001
                        s.stepsize=2
                        s.momentum = 0.9
                        s.weight_decay = 5e-4
                        s.display = 1
                        s.snapshot = 10000
                        s.snapshot_prefix = 'examples/midann/data/train'
                        s.solver_mode = caffe_pb2.SolverParameter.GPU
                        return s

                    solver_path = 'examples/midann/py_solver.prototxt'
                    with open(solver_path, 'w') as f:
                        f.write(str(solver(train_net_path, test_net_path)))
					#########################################################################################################################################


					
					############################################### Solver Mode ##########################################################################
                    caffe.set_mode_gpu()
                    solver = caffe.get_solver(solver_path)
                    solver.solve()
					########################################################################################################################################

					
					
					############################################### Initialize variable to store result ####################################################
                    rela = []
                    relb = []
                    relname = []
                    bag_label = []
					
                    #tSNE_input_data = []
                    #tSNE_output_data = []
                    #rela=np.asarray(rela)
                    #relb=np.asarray(relb)

                    accuracy = 0
                    batch_size = solver.test_nets[0].blobs['data'].num
                    test_iters = 15#int(len(test_data)/batch_size) + 1
                    print len(test_data)
                    print batch_size
                    print(test_iters)
                    print('now')
                    #test_acc = zeros(int(np.ceil(test_iters)))

                    #auc_all=[]
                    auc_sum=0
                    auc_count=0
                    #print test_iters
                    #rela=np.arange(test_iters)
                    #relb=np.arange(test_iters)
                    svm_test_feature=[]
                    svm_test_label=[]
                    svm_test_bag_label=[]
					######################################################################################################################################################
					
					
					
					########################################################### AUC and AUPRC value calculation #########################################################
                    for itt in range(test_iters):
                        solver.test_nets[0].forward()
                        #print solver.test_nets[0].blobs['ip4'].data
                        #print solver.test_nets[0].blobs['lp_label'].data

                        if(itt==0):
                            tSNE_input_data = solver.test_nets[0].blobs['data'].data
                            tSNE_output_data = solver.test_nets[0].blobs['ip2'].data
                        else:
                            tSNE_input_data=np.vstack((tSNE_input_data, solver.test_nets[0].blobs['data'].data))
                            tSNE_output_data=np.vstack((tSNE_output_data, solver.test_nets[0].blobs['ip2'].data))
                        #print solver.test_nets[0].blobs['ip4'].data
                        #print solver.test_nets[0].blobs['ip5'].data
                        rela.append(solver.test_nets[0].blobs['ip4'].data.flatten().tolist())
                        relb.append(solver.test_nets[0].blobs['lp_label'].data.flatten().tolist())
                        relname.append(solver.test_nets[0].blobs['instance_label'].data.flatten().tolist())
                        bag_label.append(solver.test_nets[0].blobs['bag_label'].data.flatten().tolist())
        #		    svm_test_feature=(solver.test_nets[0].blobs['target_feature'].data.tolist())
        #			svm_test_label=(solver.test_nets[0].blobs['lp_label'].data.tolist())
        #			svm_test_bag_label=(solver.test_nets[0].blobs['bag_label'].data.tolist()) 

                    rela=np.asarray(rela).flatten()
                    relb=np.asarray(relb).flatten()
                    relname=np.asarray(relname).flatten()
                    bag_label=np.asarray(bag_label).flatten()

                    pred_cat=np.column_stack((relname, bag_label, rela, relb))
                    print type(rela)
                    print rela 
                    print relb
                    print('..............................................')
                    pred_df=pd.DataFrame(pred_cat)
                    #print(pred_df)
                    
                    #pred_df.columns = ['isoformID', 'geneID', 'pred_label','true_label']
                    neg_pred_group=pred_df[pred_df[3]==0] #pred_df.groupby(['true_label']).get_group(0)
                    threshold=neg_pred_group[2].quantile(0.9)
                    print(neg_pred_group)
                    print(threshold)

                    sig_pred=pred_df[pred_df[1].isin(sig)]
                    sig_pred=sig_pred.drop_duplicates()	
                    #print(sig_pred)

                    mig_pred=pred_df[pred_df[1].isin(mig)]
                    mig_pred=mig_pred.drop_duplicates()
                    #print(mig_pred)	    

                    sig_rela=sig_pred[2].tolist()
                    sig_relb=sig_pred[3].tolist()

                    #mig_rela=mig_pred[2].tolist()
                    #mig_relb=mig_pred[3].tolist()

                    fpr, tpr, t = roc_curve(sig_relb, sig_rela)
                    roc_auc_midann = auc(fpr, tpr)
                    if math.isnan(roc_auc_midann)!=True:
                        midann_auc_all.append(roc_auc_midann)

                    precision, recall, thr = precision_recall_curve(sig_relb, sig_rela)
                    pr_auc_midann = auc(recall, precision)
                    if math.isnan(roc_auc_midann)!=True:
                        midann_pr_auc_all.append(pr_auc_midann)

                    mig_pred.columns = ['isoformID', 'geneID', 'pred_label','true_label']
                    idx = mig_pred.groupby(['geneID'])['pred_label'].transform(max) == mig_pred['pred_label']
                    mig_pred_max=mig_pred[idx]
                    
                    print(mig_pred.shape)
                    print(mig_pred_max.shape)
                    
                    mig_rela=mig_pred_max['pred_label'].tolist()
                    mig_relb=mig_pred_max['true_label'].tolist()

                    fpr_mig, tpr_mig, t_mig = roc_curve(mig_relb, mig_rela)
                    roc_auc_midann_mig = auc(fpr_mig, tpr_mig)
                    if math.isnan(roc_auc_midann_mig)!=True:
                        midann_auc_all_mig.append(roc_auc_midann_mig)

                    precision_mig, recall_mig, thr_mig = precision_recall_curve(mig_relb, mig_rela)
                    pr_auc_midann_mig = auc(recall_mig, precision_mig)
                    if math.isnan(roc_auc_midann_mig)!=True:
                        midann_pr_auc_all_mig.append(pr_auc_midann_mig)

                    restable_pred = pd.DataFrame(
                    {'isoformID': relname,
                    'truelabel': relb,
                    'predlabel': rela
                    })

                    final_prediction_pred=pd.concat([restable_pred,final_prediction_pred],ignore_index=True)
                    final_prediction_pred=final_prediction_pred.drop_duplicates(subset='isoformID')
                    print(final_prediction_pred)
                    final_prediction_pred=final_prediction_pred[final_prediction_pred.predlabel != 0]
                    print(final_prediction_pred)
					
    #        		 ################################### Discurd #####################################################################################
    #                #for iid in final_prediction['isoformID']:
    #                #       print(example_set[0].tolist().index(iid))
    #                #       example_set[example_set[:,1].tolist().index(iid),2]=1
    #                final_prediction_pred.to_csv('predtable.txt', sep='\t')
					 ##################################################################################################################################

                    for iid in final_prediction_pred['isoformID']:
    #                    print(example_set[:,1].tolist().index(iid))
                        example_set_copy[example_set_copy[:,1].tolist().index(iid),2]=final_prediction_pred.loc[final_prediction_pred['isoformID'] == iid, 'predlabel'].iloc[0].astype(float)                    
                    try:
                        df_pred=pd.read_csv(child_pred, sep='\t')
                    except:
                        print('Empty File. First Column about to write into it \n')
    #
                    print(GOID)
                    df_pred[GOID] = example_set_copy[:,2]    ##append predicted result column in predchild_file
                    df_pred.to_csv(child_pred, sep='\t', index=False)
					########################################################################################################################################
					
					
					
					#################################################### tSNE figure plotting ############################################################
    #                print('..............................................')
    #                print type(tSNE_input_data)
    #                print tSNE_input_data
    #                print type(tSNE_output_data)
    #                print tSNE_output_data
    #                print tSNE_input_data.shape
    #                print tSNE_output_data.shape
    #
    #
    #                x_data = np.asarray(tSNE_input_data).astype('float64')
    #                #x_data = x_data.reshape((x_data.shape[0], -1))
    #                # For speed of computation, only run on a subset
    #                #n = 1500
    #                x_data = x_data
    #                y_data = relb
    #                # perform t-SNE embedding
    #
    #        		#	    vis_data = bh_sne(x_data)
    #            	# plot the result
    #        		#	    vis_x = vis_data[:, 0]
    #        		#	    vis_y = vis_data[:, 1]
    #        		#	    plt.scatter(vis_x, vis_y, c=y_data)#, cmap=plt.cm.get_cmap("jet", 2))
    #            	##plt.colorbar(ticks=range(2))
    #            	##plt.clim(-0.5, 0.5)
    #        		#	    plt.show()
					###########################################################################################################################################
				
				

    #				 ################################################### tSNE figure plotting ############################################################
    #                x_data = np.asarray(tSNE_output_data).astype('float64')
    #                #x_data = x_data.reshape((x_data.shape[0], -1))
    #                # For speed of computation, only run on a subset
    #                #n = 1500
    #                x_data = x_data
    #                y_data = rela
    #                # perform t-SNE embedding
    #
    #    #            vis_data = bh_sne(x_data)
    #                # plot the result
    #    #            vis_x = vis_data[:, 0]
    #    #            vis_y = vis_data[:, 1]
    #    #            plt.scatter(vis_x, vis_y, c=y_data)#, cmap=plt.cm.get_cmap("jet", 2))
    #                ##plt.colorbar(ticks=range(2))
    #                ##plt.clim(-0.5, 0.5)
    #    #            plt.show()
					 ###########################################################################################################################################

					 
					 
					######################################################## final prediction after thresholding ##############################################
					rela[rela>threshold]=1
                    rela[rela<=threshold]=0
                    target_batch_size = solver.net.blobs['target_data'].num
                    target_iters = int(len(target_data)/batch_size) + 1

                    restable = pd.DataFrame(
                    {'isoformID': relname,
                    'truelabel': relb,
                    'predlabel': rela
                    })

                    final_prediction=pd.concat([restable,final_prediction],ignore_index=True)
                    final_prediction=final_prediction.drop_duplicates(subset='isoformID')
                    final_prediction=final_prediction[final_prediction.predlabel != 0]
                    #print(final_prediction)
					
					##################### Discurd ######################################################
                    #for iid in final_prediction['isoformID']:
                    #	print(example_set[0].tolist().index(iid))
                    #	example_set[example_set[:,1].tolist().index(iid),2]=1
                    #   final_prediction.to_csv('predtable.txt', sep='\t')	
					####################################################################################
					
                    for iid in final_prediction['isoformID']:
                        #print(example_set[:,1].tolist().index(iid))
                        example_set[example_set[:,1].tolist().index(iid),2]=1	
                    try:	
                        df=pd.read_csv(predchild_file , sep='\t')
                    except:
                        print('Empty File. First Column about to write into it \n')

                    print(GOID)
                    df[GOID] = example_set[:,2]    ##append predicted result column in predchild_file
                    df.to_csv(predchild_file, sep='\t', index=False)
					########################################################################################################################################
					
                
			##################################### End of all processes for a particular GO term #############################################################
            ################################################ store final result after cross validation #######################################################
            total_time=time.time() - start_time;
            print("--- %s seconds ---" % (time.time() - start_time))
            print it
            print(type(it))
            try:    
                print it
                print midann_auc_all
                print GO_term_size
                print np.mean(midann_auc_all)
                print np.mean(midann_pr_auc_all)
                
                fout_auc=open(auc_fileout,'a')
                fout_auc.write(it + '\t')
                fout_auc.write( str(neuronL1) + '\t')
                fout_auc.write( str(GO_term_size)+ '\t' )
                fout_auc.write( str(total_time) + '\n' )
                fout_auc.close()
            except:
                print(sys.exc_info()[0])   
			###################################################################################################################################################
fout_prauc.close()
fout_predchild.close()


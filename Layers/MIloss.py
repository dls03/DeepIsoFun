import sys
#import os

#caffe_root = '/rhome/dshaw003/bigdata/caffe/caffe-master/'  # this file is expected to be in {caffe_root}/examples
#sys.path.append(caffe_root + 'python')

import caffe
import numpy as np
import copy

class MultipleInstanceLossLayer(caffe.Layer):
    """
    Compute the Multiple Instance Loss in the same manner as the C++ Multiple InstanceLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        print len(bottom)
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute loss.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count and bottom[1].count != bottom[2].count :
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        #print('\n\n\n\n hoise \n\n\n\n')
        #print bottom[0].data.shape
##	self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        #self.diff.reshape(10,1)
	#print self.diff.shape
	# loss output is scalar
        #top[0].reshape(1)

    def forward(self, bottom, top):
	
	#print('calculate loss functions \n\n\n\n\n');
        bag_data=(bottom[2].data)
        pred_data=(bottom[0].data)
        label_data=(bottom[1].data)
	#t_feature=bottom[3].data
        
	

	i_size=int(bag_data.max())+1;
        bag_size=np.zeros(i_size)
        bag_id=np.zeros(i_size)
        bag_pred_prod=np.ones(i_size)
        bag_true_label=np.zeros(i_size)
        bag_pred_sum=np.zeros(i_size)   #print bottom[2].data
        
#	print '\n\n\n\n'
#        print bag_data
#        print pred_data

	j=0
        for i in bag_data:
#	    print i
            bag_id[int(i)] = int(i)
            bag_size[int(i)]=bag_size[int(i)]+1
	    bag_pred_sum[int(i)] = bag_pred_sum[int(i)]+pred_data[j]
	    bag_pred_prod[int(i)]=bag_pred_prod[int(i)]*(1-pred_data[j])
	    bag_true_label[int(i)]=label_data[j]
	    j=j+1


	delid=[]
	for i in range(len(bag_id)):
		if bag_id[i]==0:
			delid.append(i)
	
	bag_id=np.delete(bag_id,delid)
	bag_size=np.delete(bag_size,delid)
	bag_pred_sum=np.delete(bag_pred_sum,delid)
	bag_pred_prod=np.delete(bag_pred_prod,delid)
	bag_true_label=np.delete(bag_true_label, delid)


        #print type(bag_id)
        #print bag_id
        #print bag_size
	#print bag_pred_prod
	#print bag_true_label
	#bag_id=list(set(bag_id))
        #max_bag=0 
        #for i in range(len(bag_id)):
        #    if bag_id[i]!=0:
                #print('bag_id %d' % bag_id[i])
                #print('bag_size %d' %bag_size[i])
        #        max_bag=bag_id[i]
        
        #print 'max size of bag %d' %max_bag 
        

        P=[]
        Y=[]
	S=[]
	#iteri=0
        #mul=1
        #for i in bag_id:
        #    mul=1
        #    for j in range(int(bag_size[i])):
        #        mul=mul*(1-pred_data[iteri])
        #        iteri=iteri+1
        #    #print ' For Bag %d %f' % (i+1, (1-mul))
        #    P.append(1-mul) 
	
	for i in range(len(bag_id)):
		P.append(1-bag_pred_prod[i])
      		Y.append(bag_true_label[i])
		S.append(bag_pred_sum[i])
	
        #bag_data=bag_data.astype(np.int_)
        #bag_data=bag_data.flatten()
        #bag_data=bag_data.tolist()

        #print bag_data
        #print label_data



        #for i in range(int(max_bag)):
        #    Y.append(label_data[bag_data.index(i+1)])
            #print ' For Bag %d %f' % (i+1, label_data[bag_data.index(i+1)])


        #print len(pred_data)
        #print iteri
        #print '\n\n\n\n\n\n\n printing P \n\n\n\n\n'
        
	#print(P)
        #print(Y)
	#print len(P)    
        #print len(Y)
        #print pred_data.shape 
        #print sum(Y)

        #for i in range(pred_data.shape[0]-len(P)):
        #    P.append(0)
        #    Y.append(0)
       

        P=[[i] for i in P]
        Y=[[i] for i in Y]
	S=[[i] for i in S]
        #sys.exit()        
        
	
	P=np.asarray(P)
        Y=np.asarray(Y)
	S=np.asarray(S)
	


	#print(pred_data.shape)
#	print(pred_data)
	#print(label_data.shape)
#	print(label_data)
#	print '\n\n\n\n\n\n'
	#print(P.shape)
#	print(P)
	#print(Y.shape)
#	print(Y)

	top[0].reshape(P.shape[0],P.shape[1])
	top[1].reshape(Y.shape[0],Y.shape[1])

##	self.diff = np.zeros_like(P, dtype=np.float32)       
##	self.diff[...] = P - Y


	top[0].data[...]=P
	top[1].data[...]=Y
        #top[0].data[...] = np.sum(self.diff**2) / len(Y) / 2.
	#self.diff[...] = (P - Y) * S / P
	#top[0].data[...] = np.sum(Y*np.log(P)+(1-Y)*np.log(1-P))
	
	del bag_data;
	del pred_data;
	del label_data;
	del bag_size;
	del bag_id;
	del bag_pred_prod;
	del bag_true_label;
	del bag_pred_sum;


    def backward(self, top, propagate_down, bottom):

	pass	    
	    
##        for i in range(2):
##            if not propagate_down[i]:
##                continue
##            if i == 0:
##                sign = 1
##            else:
##                sign = -1
##            bottom[i].diff[...] = sign * self.diff / bottom[i].num
##	    #bottom[i].diff[...]=self.diff/bottom[i].num

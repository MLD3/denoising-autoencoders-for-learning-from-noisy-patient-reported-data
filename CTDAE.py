# Commands used to run all models:

#Enviornment set up (local to original enviornment, of course):
#source ~/venv/bin/activate;
#export CUBLAS_WORKSPACE_CONFIG=:4096:2;
#export CUDA_VISIBLE_DEVICES=X

# TUNING:
# for a in .1 .3 .5 .7 .9 1 1.25 1.5 1.75 2 ;do python unc.py 0 $a N2N tune ;done;
# for a in 500 250;do for b in .5 .333 .667;do for c in .1 .3 .5 .7;do python unc.py 0 z $a $b $c  COTEACH tune; done;done;done;
# for a in 500 250;do for b in .5 .333 .667;do for c in .1 .3 .5 .7;do python unc.py 0 z $a $b $c  SUPCT tune; done;done;done;

# Main exps:
# for b in 0 1 2 3 4 5 6 7 8 9;do for c in CAE NAE COTEACHP N2N SUP SUPCT;do python unc.py $b $c;done;done;  

# Alt noises:
# for a in CAE NAE COTEACHP N2N SUP SUPCT;do for s in 0 1 2 3 4 5 6 7 8 9 ;do for t in 1 2 3 4 5 6 7;do python unc.py $s $t $a VARSALT;done;done;done ;

# REAL DATA ANALYSES
# for s in 0 1 2 3 4 5 7 10;do for a in NAE COTEACHP N2N SUP SUPCT;do python unc.py $s $a $b OHIO;done;done


import os,datetime
import sys
import matplotlib.pyplot as plt
import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
import joblib
import numpy as np
import pandas as pd
import shutil 
from scipy.ndimage import gaussian_filter1d as gaus
import scipy.special as ss
torch.autograd.set_detect_anomaly(True)
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"	 

def inarg(i,o):
	if i in sys.argv:
		return True,o+'-'+i
	return False,o

############################################## Training/model params
SEED=1
BATCHSIZE = 500
LONGMIN=500 #minimum number of iterations to run
PATIENCE=50
UNITS=100 #LSTM size


############################################ data params
nv=4 #number of variables in model
NOMISS=False #remove missing glucose windows (real data)
NODOUBS=False #remove windows with >1 carb value (real data)
CF=True #carry forward carb and bolus values
input_len=24 #timepoints input
horizon=12 #for forecasting, not in CTDAE paper
datalen=100 #number of days used for simulated data training
CARBO=False #CARBO=True indicates that zero-valued carbs are not used.


################################### Subject and model selection



####Simulated data section###

outstr='SIMGLU'

#vary amount of data used, data noise type, or amount of time used for input
VARD=False
VARS=False
VART=False
datalen=100
if not 'OHIO' in sys.argv:
	subs=['01','02','03','04','05','06','07','08','09','10']
	
	outstr+='-SUB'+sys.argv[1]
	pickedsub=subs[int(sys.argv[1])]
	VART,outstr=inarg('VART',outstr)
	VARD,outstr=inarg('VARD',outstr)
	VARS,outstr=inarg('VARSALT',outstr)
	
	if VARS:
		noisetype=int(sys.argv[2])
	if VART or VARD or VARS:
		outstr+=sys.argv[2]
	if VART:
		input_len=int(sys.argv[2])
	if VARD:
		datalen=int(sys.argv[2])
	loadstr='/data3/interns/unc/adult'+pickedsub+'-'
	if VARS:
		loadstr+=sys.argv[2]+'ALT'
	else:
		loadstr+='0'
	loadstr+='.pkl'
	print('loading ',loadstr)



##### real data section #####
su=''
OHIO,outstr=inarg('OHIO',outstr)
if OHIO:
	NOMISS=True
	NODOUBS=True
	subs=['559','563','570','575','588','591','540','544','552','584','596','567']
	su=subs[int(sys.argv[1])]
	outstr=su



	#load in training data
	a=joblib.load('/data3/interns/postohio/allohiodata/'+su+'.train.pkl')
	g=np.asarray(a['glucose'])/400
	b=np.asarray(a['basal'])/50
	d=np.asarray(a['dose'])/50
	c=np.asarray(a['carbs'])/200
	g[np.isnan(g)]=0
	b[np.isnan(b)]=0
	c[np.isnan(c)]=0
	d[np.isnan(d)]=0
	c[c>1]=0 #get rid of that 588 outlier
	ALLDAT=np.stack((g,d,c,b,b*0),axis=1)


	#grab carb and non carbwindows
	trainvaldatall=[]
	trainvaldataNC=[]
	for i in range(ALLDAT.shape[0]-36):
		if ALLDAT[i,2]>0:
			for j in [i+1,i+2,i+3]:
				ALLDAT[i,2]+=ALLDAT[j,2]
				ALLDAT[j,2]=0
			temp=ALLDAT[i:i+24+horizon,:]
			if np.sum(temp[i+1:i+24,2])>0:
				continue
			if np.product(temp[:,0])==0:
				continue
			trainvaldatall.append(temp)
		else:
			temp=ALLDAT[i:i+24+horizon,:]
			if np.product(temp[:,0])!=0 and np.sum(temp[:24,2])==0:
				trainvaldataNC.append(temp)

	#report number of training carbs
	ALLCARBS=np.array(trainvaldatall)[:,0,2]
	print(su,len(ALLCARBS))

	#load in test data
	a=joblib.load('/data3/interns/postohio/allohiodata/'+su+'.test.pkl')
	g=np.asarray(a['glucose'])/400
	b=np.asarray(a['basal'])/50
	d=np.asarray(a['dose'])/50
	c=np.asarray(a['carbs'])/200
	g[np.isnan(g)]=0
	b[np.isnan(b)]=0
	c[np.isnan(c)]=0
	d[np.isnan(d)]=0
	TESTDAT=np.stack((g,d,c,b,b*0),axis=1)
	testdat=[]
	testdatNC=[]
	for i in range(TESTDAT.shape[0]-36):
		if TESTDAT[i,2]>0:
			for j in [i+1,i+2,i+3]:
				TESTDAT[i,2]+=TESTDAT[j,2]
				TESTDAT[j,2]=0
			temp=TESTDAT[i:i+24+horizon,:]
			if np.sum(temp[i+1:i+24,2])>0:
				continue
			if np.product(temp[:,0])==0:
				continue
			testdat.append(temp)
		else:
			temp=TESTDAT[i:i+24+horizon,:]
			if np.product(temp[:,0])!=0 and np.sum(temp[:24,2])==0:
				testdatNC.append(temp)




#model selection
CAE,outstr=inarg('CAE',outstr)
NAE,outstr=inarg('NAE',outstr)
N2N,outstr=inarg('N2N',outstr)
SUP,outstr=inarg('SUP',outstr)
SUPCT,outstr=inarg('SUPCT',outstr)
COTEACH,outstr=inarg('COTEACH',outstr)
COTEACHP,outstr=inarg('COTEACHP',outstr)
if COTEACHP or SUPCT:
	COTEACH=True
if SUPCT:
	SUP=True
if SUP:
	NAE=True


#Tune or set hyper-params
nalph=1
if N2N:
	NAE=True
	if 'tune' in sys.argv:
		nalph=float(sys.argv[2])
		outstr+=sys.argv[2]
if COTEACH:
	NAE=True
	Ek=500
	tau=.5
	CTTHRESH=0
	if SUPCT:
		Ek=500
		tau=.5
		CTTHRESH=.3
	if COTEACHP:
		Ek=250
		tau=.333
		CTTHRESH=.1
	if 'tune' in sys.argv:
		Ek=float(sys.argv[3])
		outstr+='-'+sys.argv[3]
		tau=float(sys.argv[4])
		outstr+='-'+sys.argv[4]
		CTTHRESH=float(sys.argv[5])
		outstr+='-'+sys.argv[5]

#do 2 networks for fairness
if not COTEACH:
	DOUBNET=True

ADDCGM_NOISE,outstr=inarg('ADDCGM_NOISE',outstr)
if ADDCGM_NOISE:
	NOISEAMT=float(sys.argv[2])
	outstr+='_'+sys.argv[2]

#evaluate already run model
EVALONLY= ('eval' in sys.argv) 
#################################### MAIN SECTION ############################################
def main():
	maindir = os.getcwd()+'/'+outstr
	if not EVALONLY:
		os.makedirs(maindir)
	np.random.seed(SEED)
	torch.manual_seed(SEED)
	train_and_evaluate(maindir,horizon,input_len)




####################################	TRAINING, AND EVALUATION SECTION ############################################
def train_and_evaluate(mydir,forecast_length,backcast_length):

	#Set up data
	batch_size = BATCHSIZE
	#main data set
	train,val,test=makedata(backcast_length+forecast_length)
	traingen = data(333+CARBO*167, backcast_length, forecast_length,train)
	valgen = data(333+CARBO*167, backcast_length, forecast_length,val)
	testgen = ordered_data(batch_size, backcast_length, forecast_length,test)
	#evaluation datasets
	traintestgen=ordered_data(batch_size, backcast_length, forecast_length,train)
	valtestgen=ordered_data(batch_size, backcast_length, forecast_length,val)
	#clean data for reference
	traindX,valttX,testt=makedata(backcast_length+forecast_length,True)
	testtru=ordered_data(batch_size, backcast_length, forecast_length,testt)
	valtru=ordered_data(batch_size, backcast_length, forecast_length,valttX)
	#data where Y=0 (no carb data)
	trainNC,valNC,testNC=makedata(backcast_length+forecast_length,False,True)
	traingenNC = data(167, backcast_length, forecast_length,trainNC)
	valgenNC = data(167, backcast_length, forecast_length,valNC)
	

	#set up networks
	pin_memory=True
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	net = network(device,backcast_length,forecast_length)
	optimiser = optim.Adam(net.parameters(),lr=.01,weight_decay=.0000001)
	net2=None
	optimiser2=None
	if COTEACH or DOUBNET:
		net2 = network(device,backcast_length,forecast_length)
		optimiser2 = optim.Adam(net2.parameters(),lr=.01,weight_decay=.0000001)

	#Train the network
	if not EVALONLY:
		if CARBO:
			fit(net, optimiser, traingen,valgen,mydir, device,None,None,net2, optimiser2)
		else:
			fit(net, optimiser, traingen,valgen,mydir, device,traingenNC,valgenNC,net2, optimiser2)


	#run various evaluations
	eval(net, optimiser, testgen,mydir,	device,'',testtru,net2, optimiser2)
	eval(net, optimiser, valtestgen,mydir,	device,'val',valtru,net2, optimiser2)
	eval(net, optimiser, testtru,mydir,	device,'tru',testtru,net2, optimiser2)
	
	





def fit(net, optimiser, traingen,valgen,mydir,device, NCDT,NCDV,net2,optimiser2):
	#initialize tracking variables and networks
	improvepoint=0
	trains=[]
	vals=[]
	patience=PATIENCE
	prevvalloss=np.inf
	unimproved=0
	net.to(device)
	if COTEACH or DOUBNET:
		net2.to(device)

	for grad_step in range(5000000):
		if COTEACH:
			#calculate co-teaching sample size.
			CTITTHRESH=1-np.minimum((grad_step+1)/Ek*tau,tau)

		temptrain=[]
		total=0
		while(True):
			optimiser.zero_grad()
			net.train()
			if COTEACH or DOUBNET:
				optimiser2.zero_grad()
				net2.train()
			#set up data
			x,targ,done=next(traingen)
			if x.shape[0]<1:
				break
			if NCDT is not None:
				xNC,targNC,doneNC=next(NCDT)
				if xNC.shape[0]<1:
					xNC,targNC,doneNC=next(NCDT)
				x=np.concatenate((x,xNC),0)
			
			
			
			#set up target
			if NAE:
				temptc=x[:,0,2].copy()#noisy reference values
			elif CAE:
				temptc=x[:,0,4].copy()#true  reference values
				x[:,0,2]=temptc.copy()#use true reference values as input
			#indices where carbs are or are not 0
			inds0=x[:,0,2]==0
			indsg0=x[:,0,2]>0
			xorigg=x.copy()
			xorigg=xorigg[:,:,:4]
			#add noise (Y-->Z)
			x[:,0,2]*=(1+np.random.normal(0,.5*nalph,size=x[:,0,2].shape))
			x[:,0,2][x[:,0,2]<0]=0
			x[:,0,2][x[:,0,2]>1]=1
			x[:,0,2]*=1*(np.random.uniform(size=x[:,0,2].shape)<.5)
			if SUP: #if supervised, don't use Z
				x[:,0,2]=0
			#clean up inputs
			x=x[:,:,:4]
			xin=x.copy()
			if x.shape[0]<1:
				continue
			
			#run network
			carbup=	net(	 torch.tensor(xin, dtype=torch.float).to(device))
			if COTEACH or DOUBNET:
				carbup2=	net2(	 torch.tensor(xin, dtype=torch.float).to(device))
			total=total+x.shape[0]
			

			
			if COTEACH:
				#calculate updated carb values (these are Y_hat) in a gradient-friendly way.
				carbuptemp=carbup.clone()
				carbuptemp2=carbup2.clone()
				carbup=carbuptemp.clone()+torch.tensor(x[:,0,2], dtype=torch.float).to(device)
				carbup2=carbuptemp2.clone()+torch.tensor(x[:,0,2], dtype=torch.float).to(device)
				carbup[carbup<0]*=0
				carbup2[carbup2<0]*=0

				#initialize losses
				loss=0
				loss2=0


				#assuming that we are including 0 valued carbs,
				#add the squared error corresponding to them to the loss function
				#and remove from co-teaching sample.
				if NCDT is not None:
					loss=torch.mean(carbup[inds0]**2)
					loss2=torch.mean(carbup2[inds0]**2)
					carbup=carbup[indsg0]
					carbup2=carbup2[indsg0]
					temptc=temptc[indsg0]
					xinb=xorigg.copy()[indsg0]
					
				#calculate mean squared percent error between model outputs for co-teaching+
				mspe=np.abs((carbup2.detach().cpu().numpy()-carbup.detach().cpu().numpy())/temptc)*100
				#remove the samples for which the two models have the most similar results (co-teaching+ variant)
				carbup=carbup[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				carbup2=carbup2[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				temptc=temptc[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				xinc=xinb.copy()[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				temptc2=temptc.copy()
				fullsize=carbup.shape[0]
				
				#if we are in a supervised setting, no Y information goes to the model
				if SUP:
					xinc[:,0,2]=0
				#calculate carb updates for clean Y values (to generate Y_tilde values)
				carbuprawt=	net(	 torch.tensor(xinc, dtype=torch.float).to(device))
				carbup2rawt=	net2(	 torch.tensor(xinc, dtype=torch.float).to(device))
				
				
				#we need the original carb values for later.
				comper=xinc[:,0,2]
				if SUP:
					comper=temptc.copy()

				#calculate Y_tilde values in a gradient friendly way
				carbupraw=carbuprawt.clone()+torch.tensor(xinc[:,0,2], dtype=torch.float).to(device)
				carbup2raw=carbup2rawt.clone()+torch.tensor(xinc[:,0,2], dtype=torch.float).to(device)
				carbupraw[carbupraw<0]*=0
				carbup2raw[carbup2raw<0]*=0
				#calculate MSPE values for Y_tilde values
				rawcarberr1=( (carbupraw.detach().cpu().numpy()-comper)/comper)**2
				rawcarberr2=( (carbup2raw.detach().cpu().numpy()-comper)/comper)**2

				#remove samples with highest Y_tilde_1 MSPE values from DAE2 backprop sample and vice versa
				carbup=carbup[np.argsort(rawcarberr2)][:int(fullsize*CTITTHRESH)]
				carbup2=carbup2[np.argsort(rawcarberr1)][:int(fullsize*CTITTHRESH)]
				temptc=temptc[np.argsort(rawcarberr2)][:int(fullsize*CTITTHRESH)]
				temptc2=temptc2[np.argsort(rawcarberr1)][:int(fullsize*CTITTHRESH)]
				
				#calculate loss using only the selected sample.
				loss+=torch.mean((carbup- torch.tensor(temptc, dtype=torch.float).to(device))**2)
				loss2+=torch.mean((carbup2- torch.tensor(temptc2, dtype=torch.float).to(device))**2)

			else:
				#calculate updated carb values
				if DOUBNET:
					ncc2=torch.tensor(np.max(x[:,:,2],1)).to(device)+carbup2
					ncc2[ncc2<0]*=0
				ncc=torch.tensor(np.max(x[:,:,2],1)).to(device)+carbup
				ncc[ncc<0]*=0
				#calculate loss.
				loss=torch.mean((ncc- torch.tensor(temptc, dtype=torch.float).to(device))**2)
				if DOUBNET:
					loss2=torch.mean((ncc2- torch.tensor(temptc, dtype=torch.float).to(device))**2)
			#final steps of train loop.
			loss.backward()
			optimiser.step()
			if COTEACH or DOUBNET:
				loss2.backward()
				optimiser2.step()
			temptrain.append(loss.item()*x.shape[0])
			if done:
				break

		
		
		trains.append(np.sum(temptrain)/total)
		print('grad_step = '+str(grad_step)+' loss = '+str(trains[-1]))
		
		
		tempval=[]
		total=0
		while(True):
			#validation step for early stopping.
			with torch.no_grad():
				#set up data
				x,target,done=next(valgen)
				if x.shape[0]<1:
					break
				if NCDV is not None:
					xNC,targNC,doneNC=next(NCDV)
					if xNC.shape[0]<1:
						xNC,targNC,doneNC=next(NCDV)
					x=np.concatenate((x,xNC),0)
				#set up target
				if NAE:
					temptc=x[:,0,2].copy()
				elif CAE:
					temptc=x[:,0,4].copy()
					x[:,0,2]=temptc.copy()



				#indices where carbs are or are not 0
				inds0=x[:,0,2]==0
				indsg0=x[:,0,2]>0
				xorigg=x.copy()
				xorigg=xorigg[:,:,:4]
				#add noise (Y-->Z)
				x[:,0,2]*=(1+np.random.normal(0,.5*nalph,size=x[:,0,2].shape))
				x[:,0,2][x[:,0,2]<0]=0
				x[:,0,2][x[:,0,2]>1]=1
				x[:,0,2]*=1*(np.random.uniform(size=x[:,0,2].shape)<.5)
				if COTEACH or SUP: #if supervised or co-teach, don't use Z.
					x[:,0,2]=0
				#clean up inputs
				x=x[:,:,:4]
				xin=x.copy()
				if x.shape[0]<1:
					continue

				#calculate updates.
				carbup=	net(	 torch.tensor(xin, dtype=torch.float).to(device))		
				if COTEACH or DOUBNET:
					carbup2=	net2(	 torch.tensor(xin, dtype=torch.float).to(device))

			
			
			if COTEACH:
				#calculate updated carb values (these are Y_hat) in a gradient-friendly way.
				carbuptemp=carbup.clone()
				carbuptemp2=carbup2.clone()
				carbup=carbuptemp.clone()+torch.tensor(x[:,0,2], dtype=torch.float).to(device)
				carbup2=carbuptemp2.clone()+torch.tensor(x[:,0,2], dtype=torch.float).to(device)
				carbup[carbup<0]*=0
				carbup2[carbup2<0]*=0

				#initialize losses
				loss=0
				loss2=0

				#assuming that we are including 0 valued carbs,
				#add the squared error corresponding to them to the loss function
				#and remove from co-teaching sample.
				if NCDT is not None:
					loss=torch.mean(carbup[inds0]**2)
					loss2=torch.mean(carbup2[inds0]**2)
					carbup=carbup[indsg0]
					carbup2=carbup2[indsg0]
					temptc=temptc[indsg0]
					xinb=xorigg.copy()[indsg0]
					
				#calculate mean squared percent error between model outputs for co-teaching+
				mspe=np.abs((carbup2.detach().cpu().numpy()-carbup.detach().cpu().numpy())/temptc)*100
				#remove the samples for which the two models have the most similar results (co-teaching+ variant)
				carbup=carbup[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				carbup2=carbup2[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				temptc=temptc[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				xinc=xinb.copy()[np.argsort(mspe)][int(len(mspe)*CTTHRESH):]
				temptc2=temptc.copy()
				fullsize=carbup.shape[0]
				
				#if we are in a supervised setting, no Y information goes to the model
				if SUP:
					xinc[:,0,2]=0
				#calculate carb updates for clean Y values (to generate Y_tilde values)
				carbuprawt=	net(	 torch.tensor(xinc, dtype=torch.float).to(device))
				carbup2rawt=	net2(	 torch.tensor(xinc, dtype=torch.float).to(device))
				
				
				#we need the original carb values for later.
				comper=xinc[:,0,2]
				if SUP:
					comper=temptc.copy()

				#calculate Y_tilde values in a gradient friendly way
				carbupraw=carbuprawt.clone()+torch.tensor(xinc[:,0,2], dtype=torch.float).to(device)
				carbup2raw=carbup2rawt.clone()+torch.tensor(xinc[:,0,2], dtype=torch.float).to(device)
				carbupraw[carbupraw<0]*=0
				carbup2raw[carbup2raw<0]*=0
				#calculate MSPE values for Y_tilde values
				rawcarberr1=( (carbupraw.detach().cpu().numpy()-comper)/comper)**2
				rawcarberr2=( (carbup2raw.detach().cpu().numpy()-comper)/comper)**2

				#remove samples with highest Y_tilde_1 MSPE values from DAE2 backprop sample and vice versa
				carbup=carbup[np.argsort(rawcarberr2)][:int(fullsize*CTITTHRESH)]
				carbup2=carbup2[np.argsort(rawcarberr1)][:int(fullsize*CTITTHRESH)]
				temptc=temptc[np.argsort(rawcarberr2)][:int(fullsize*CTITTHRESH)]
				temptc2=temptc2[np.argsort(rawcarberr1)][:int(fullsize*CTITTHRESH)]
				
				#calculate loss using only the selected sample.
				loss+=torch.mean((carbup- torch.tensor(temptc, dtype=torch.float).to(device))**2)
				loss2+=torch.mean((carbup2- torch.tensor(temptc2, dtype=torch.float).to(device))**2)

			else:
				#for validation, calculate loss based on average correction learned.
				if COTEACH or DOUBNET:
					ncc=torch.tensor(np.max(x[:,:,2],1)).to(device)+(carbup+carbup2)/2
				else:
					ncc=torch.tensor(np.max(x[:,:,2],1)).to(device)+carbup
				ncc[ncc<0]*=0
				loss=torch.mean((ncc- torch.tensor(temptc, dtype=torch.float).to(device))**2)

			total=total+x.shape[0]
			tempval.append(loss.item()*x.shape[0])
			if done:
				break
		vals.append(np.sum(tempval)/total)
		
		print('val loss: '+str(vals[-1]))				
		
		
		if vals[-1]<prevvalloss:
			#keep going if validation loss improved, and save the models.
			print('loss improved')
			improvepoint=grad_step
			prevvalloss=vals[-1]
			unimproved=0
			save(net, optimiser, grad_step,mydir)
			if COTEACH or DOUBNET:
				save(net2, optimiser2, grad_step,mydir,2)
		else:
			#track if loss not improving.
			unimproved+=1
			print('loss did not improve for '+str(unimproved)+'th time')
		#stop only if ending criteria are met.
		if (unimproved>patience) or prevvalloss<1e-7 or grad_step>1500:
			if grad_step<LONGMIN and grad_step<1500:
				continue
			print('Finished.')
			t=open(mydir+"/"+str(improvepoint)+'_ITS',"w")
			break
	#save loss plot.
	plt.plot(range(len(trains)-1),trains[1:],'k--', range(len(trains)-1),vals[1:],'r--')
	plt.legend(['train','val'])
	plt.savefig(mydir+"/loss_over_time.png")
	plt.clf()


	del net


def eval(net, optimiser, testgen,mydir,	device,OSTR,testtru,net2,optimiser2):
	with torch.no_grad():
		load(net, optimiser,mydir)
		if COTEACH or DOUBNET:
			load(net2, optimiser2,mydir,2)
		#initialize output arrays and counts
		xs=[]
		cs=[]
		ccount=0
		ce0=0
		ce1=0
		
		while(True):
			x,target,done=next(testgen)
			if x.shape[0]<1:
				break
			if SUP:
				x[:,:,2]=0 #don't use carb if supervised
			x=x[:,:,:4]

			xin=x.copy()
			carbup= net(torch.tensor(xin, dtype=torch.float).to(device))		
			if COTEACH or DOUBNET:
				carbup2=	net2(	 torch.tensor(xin, dtype=torch.float).to(device))

			xs.append(x)

			if testtru!=None:
				ccount+=x.shape[0]
				xtru,target,done=next(testtru)
				#calculate MSE of noisy sample
				ce0+=np.sum((np.max(x[:,:,2],1)*200-np.max(xtru[:,:,2],1)*200)**2)

				if COTEACH or DOUBNET:
					xttt=x.copy()[:,0,2]+(carbup.cpu().numpy()+carbup2.cpu().numpy())/2
				else:
					xttt=x.copy()[:,0,2]+carbup.cpu().numpy()
				xttt=np.maximum(xttt,0/200)


				if N2N:
					#do transform
					xttt=((1+nalph**2)*xttt-x[:,0,2])/nalph**2
					xttt[xttt<0]=0

				#calculate MSE of updated carb
				ce1+=np.sum((xttt*200-np.max(xtru[:,:,2],1)*200)**2)

			#save carb values
			if COTEACH or DOUBNET:
				cs.append( (carbup.cpu().numpy()+carbup2.cpu().numpy())/2)
			else:
				cs.append(carbup.cpu().numpy())
			if done:
				break

		#dump outputs
		if testtru!=None  and ccount>0:
			t=open(mydir+"/"+str(ce0/ccount)+'carboff'+OSTR+'0','w')
			t=open(mydir+"/"+str(ce1/ccount)+'carboff'+OSTR+'1','w')
		joblib.dump(cs,mydir+'/cs'+OSTR+'.pkl')
		joblib.dump(xs,mydir+'/xs'+OSTR+'.pkl')







###################################SAVE AND LOAD FUNCTIONS
def save(model, optimiser, grad_step,mdir,second=0):
	exstr=''
	if second>0:
		exstr='2'
	torch.save({
		'grad_step': grad_step,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimiser.state_dict(),
	}, mdir+'/model_out'+exstr+'.th')

def load(model, optimiser,mdir,second=0):
	exstr=''
	if second>0:
		exstr='2'
	if os.path.exists(mdir+'/model_out'+exstr+'.th'):
		print('loading '+mdir+' '+exstr)
		checkpoint = torch.load(mdir+'/'+'model_out'+exstr+'.th')
		model.load_state_dict(checkpoint['model_state_dict'])
		optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
		grad_step = checkpoint['grad_step']

def loadnoopt(model, optimiser,mdir,second=0):
	exstr=''
	if second>0:
		exstr='2'
	if os.path.exists(mdir+'/model_out'+exstr+'.th'):
		print('loading '+mdir)
		checkpoint = torch.load(mdir+'/'+'model_out'+exstr+'.th')
		model.load_state_dict(checkpoint['model_state_dict'])






####################################	MODEL SECTION	############################################################################################################	
class Block(nn.Module):

	def __init__(self, units, device, backcast_length, forecast_length):
		super(Block, self).__init__()
		self.backlen=backcast_length
		self.forecast_length=forecast_length
		self.input=nv
		self.device = device
		self.units=UNITS
		self.bs=BATCHSIZE
		

		#NOTE: these networks are not used but are still initialized for reproducability purposes (they were part of a forecasting model used in tandem)
		#(If you are not trying to replicate previous results EXACTLY, feel free to remove them)
		self.lstm=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
		self.dec=nn.LSTM(self.units*2,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
		self.lin=nn.Linear(self.units *2, 1).to(device)

		#The two networks used by the model
		self.lstmDAE=nn.LSTM(self.input,self.units, num_layers=2,batch_first=True,bidirectional=True).to(device)
		self.linDAE=nn.Linear(self.units *2, 1).to(device)


		#Also not used, included for reproducability, feel free to delete if you're not trying to get the exact same reported results.
		self.linglu=nn.Linear(self.units *2, input_len).to(device)
		
		self.to(device)

	


	def forward(self, xt):
		x=xt.clone()
		origbs=x.size()[0]
		#pad the unput
		if origbs<self.bs:
			x=F.pad(input=x, pad=( 0,0,0,0,0,self.bs-origbs), mode='constant', value=0).to(self.device)
		#pass the input through the LSTM and then the FC
		xin=x.clone()	
		cc0,(dum1,dum2)=self.lstmDAE(xin)
		cc=self.linDAE(cc0[:,-1,:].view((BATCHSIZE,1,-1))).view(-1)
		cc=cc*.01 #for initialization to be close to 0
		cc=cc[:origbs]
		return cc




class network(nn.Module):
	def __init__(self,device,backcast_length,forecast_length):
		super(network, self).__init__()
		self.forecast_length = forecast_length
		self.backcast_length = backcast_length
		self.hidden_layer_units = 512
		self.nb_blocks_per_stack = 1
		
		self.device=device

		self.mainblock=Block(self.hidden_layer_units,device, backcast_length, forecast_length).to(device)
		self.to(self.device)

	

	def forward(self, xx):
		x=xx.clone()
		if CF: #carry forward carb and bolus values for higher gradient impact
			for f in range(1,x.shape[1]):
				x[:,f,1:3]+=x[:,f-1,1:3]
		return self.mainblock(x)





####################################	DATA GENERATION SECTION	############################################################################################################	

def makedata(totallength,nonoise=False,bonusset=False):
		train=[]
		val=[]
		test=[]


		if OHIO:
			if bonusset: #zero only carb values
				train=trainvaldataNC[:int(len(trainvaldataNC)*.8)]
				val=trainvaldataNC[int(len(trainvaldataNC)*.8):]
				test=testdatNC
			else: #positive carb values
				train=trainvaldatall[:int(len(trainvaldatall)*.8)]
				val=trainvaldatall[int(len(trainvaldatall)*.8):]
				test=testdat
			return train,val,test


		#load and set up data
		a=joblib.load(loadstr)
		print('loading '+loadstr)
		a=np.array(a)
		atest=a[-50:,:].copy()
		a=a[:datalen,:]
		a=np.concatenate((a,atest))
		ll=datalen


		np.random.seed(1)#kept for exact reproducability; no longer needed otherwise
		noncarbone=0 #to limit number of zero value carbs included
		for f in range(ll+50):
			ff=a[f]

			
			noisyver=ff[:,4].copy()#copy over the noisy value so that noisy and clean samples match later
			if nonoise:
				ff[:,4]=ff[:,2].copy()#if this is the no noise version, set both true and noisy carbs to the true value
			else:
				#if this is a noisy version, we switch the clean and noisy values
				#(in the pkl files, index 2= clean and index 4=noisy, but in training we assume the opposite.)
				temptc=ff[:,2].copy()
				ff[:,2]=ff[:,4].copy()
				ff[:,4]=temptc.copy()

			#scale values
			ff[:,0]/=400
			ff[:,1]/=50
			ff[:,2]/=200
			ff[:,4]/=200

			if ADDCGM_NOISE:
				noise_generator = CGMNoise(f)
				
				for i in range(288):
					ff[:,0]+=next(noise_generator)*NOISEAMT/400
			#loop through timepoints
			for i in range(ff.shape[0]-input_len-horizon):
				#if we aren't generating the zero-value dataset (bonusset), add found carbs
				if noisyver[i]!=0 and not bonusset: 
					j=i
					if f<=ll*.8:
						train.append(ff[j:j+input_len+horizon,:].copy())
					elif f<ll:
						val.append(ff[j:j+input_len+horizon,:].copy())
					else:
						test.append(ff[j:j+input_len+horizon,:].copy())
				#if we are doing the zero-value dataset, add zero-value carbs, but only up to 5000 as more would be overkill.
				elif np.sum(ff[i:i+input_len,2])==0  and (np.sum(ff[i+1:i+input_len,4])==0) and bonusset and (noncarbone<5000):
					j=i
					if noncarbone<=.7*5000: #we don't need too many 0 valued carbs
						train.append(ff[j:j+input_len+horizon,:].copy())
						noncarbone+=1
					elif noncarbone<=.85*5000 :
						val.append(ff[j:j+input_len+horizon,:].copy())
						noncarbone+=1
					else:
						noncarbone+=1
						test.append(ff[j:j+input_len+horizon,:].copy())
		return train,val,test









def data(num_samples, backcast_length, forecast_length, data):
		def get_x_y(ii):	
				temp=data[0]
				done=False
				startnum=0

				for s in range(len(data)):
						temp=data[s]
						if len(temp)<backcast_length+ forecast_length+startnum:
								continue
						if ii+startnum<=len(temp)-backcast_length-forecast_length:
								done=True
								break
						ii=ii-(len(temp)-backcast_length-forecast_length-startnum)-1
				if not done:
						return None,None,True
								


				i=ii+startnum
				learn=temp[i:i+backcast_length]
				see=temp[i+backcast_length:i+backcast_length+forecast_length]
				see[np.isnan(see)]=0
				learn[np.isnan(learn)]=0
				origlearn=learn.copy()
				origsee=see.copy()


				see=temp[i+backcast_length:i+backcast_length+forecast_length]

				

				if np.prod(see[:,0])==0:
					return np.asarray([]),None,False
				if NOMISS and np.prod(learn[:,0])==0:
					return np.asarray([]),None,False
				if NODOUBS and len(learn[:,2][learn[:,2]>0])>1:
					return np.asarray([]),None,False
				if np.max(learn[:,2])>100:
					return np.asarray([]),None,False
				if np.sum(learn[:,0])==0:
					return np.asarray([]),None,False
				return learn,see,False
			 
		
		
		def gen():
				done=False
				indices=range(99999999)
				xx = []
				yy = []
				i=0
				added=0
				unset=True
				while(True):
						x, y,done = get_x_y(indices[i])
						i=i+1
						if done or i==len(indices):
								if x is not None:
									if not x.shape[0]==0:
										xx.append(x)
										yy.append(y)
								xx=np.array(xx)
								
								yield xx, np.array(yy),True
								done=False
								del xx,yy
								xx = []
								yy = []

								if unset:
										indices=np.random.permutation(i-1)
										unset=False
								else:
										indices=np.random.permutation(len(indices))
								i=0
								added=0
								continue
						if not x.shape[0]==0:
								xx.append(x)
								yy.append(y)
								added=added+1
								if added%num_samples==0:

										xx=np.array(xx)
										
										yield xx, np.array(yy),done
										del xx,yy
										xx = []
										yy = []
		return gen()



def ordered_data(num_samples, backcast_length, forecast_length, dataa):
	def get_x_y(i):	
		temp=dataa[0]
		done=False
		for s in range(len(dataa)):
			temp=dataa[s]
			#if this time series is too short, skip it.
			if len(temp)<backcast_length+ forecast_length:
				continue
			#if this index falls within this time series, we can return it
			if i<=len(temp)-backcast_length-forecast_length:
				done=True
				break
			#otherwise subtract this subject's points and keep going.
			i=i-(len(temp)-backcast_length-forecast_length)-1
		#if we're out of data, quit.
		if not done:
			return None,None,True
		learn=temp[i:i+backcast_length]
		see=temp[i+backcast_length:i+backcast_length+forecast_length]
		see[np.isnan(see)]=0
		learn[np.isnan(learn)]=0
		origlearn=learn.copy()

		see=temp[i+backcast_length:i+backcast_length+forecast_length]
		

		#only use data where the point we're trying to predict is there.
		if see[-1,0]==0:
			return np.asarray([]),None,False
		if NOMISS and (np.prod(learn[:,0])==0 or np.prod(see[:,0])==0):
			return np.asarray([]),None,False
		if NODOUBS and len(learn[:,2][learn[:,2]>0])>1:
			return np.asarray([]),None,False
		if np.max(learn[:,2])>100:
			return np.asarray([]),None,False
		return learn,see,False
	
	
	
	def gen():
		done=False
		xx = []
		yy = []
		i=0
		added=0
		while(True):
			x, y,done = get_x_y(i)
			i=i+1
			if done:
				xx=np.array(xx)
				
				yield xx, np.array(yy),True
				done=False
				xx = []
				yy = []
				i=0
				added=0
				continue
			if not x.shape[0]==0:
				xx.append(x)
				yy.append(y)
				added=added+1
				if added%num_samples==0:
					xx=np.array(xx)
					
					yield xx, np.array(yy),False
					xx = []
					yy = []
	return gen()
	

from scipy.interpolate import interp1d
import math
from collections import deque
import logging



def johnson_transform_SU(xi, lam, gamma, delta, x):
	return xi + lam * np.sinh((x - gamma) / delta)


class CGMNoise(object):
	PRECOMPUTE = 10  # length of pre-compute noise sequence
	MDL_SAMPLE_TIME = 15

	def __init__(self,seed ):
		self.seed = seed
		# self._noise15_gen = self._noise15_generator()
		self._noise15_gen = noise15_iter()
		self._noise_init = next(self._noise15_gen)

		self.n = np.inf
		self.count = 0
		self.noise = deque()

	def _get_noise_seq(self):
		# To make the noise sequence continous, keep the last noise as the
		# beginning of the new sequence
		noise15 = [self._noise_init]
		noise15.extend([next(self._noise15_gen)
						for _ in range(self.PRECOMPUTE)])
		self._noise_init = noise15[-1]

		noise15 = np.array(noise15)
		t15 = np.array(range(0, len(noise15))) * self.MDL_SAMPLE_TIME

		nsample = int(math.floor(
			self.PRECOMPUTE * self.MDL_SAMPLE_TIME / 5)) + 1
		t = np.array(range(0, nsample)) * 5

		interp_f = interp1d(t15, noise15, kind='cubic')
		noise = interp_f(t)
		noise2return = deque(noise[1:])

		# logger.debug('New noise sampled every 15 min:\n{}'.format(noise15))
		# logger.debug('New noise sequence:\n{}'.format(noise2return))

		# plt.plot(t15, noise15, 'o')
		# plt.plot(t, noise, '.-')
		# plt.show()

		return noise2return

	def __iter__(self):
		return self

	def __next__(self):
		if self.count < self.n:
			if len(self.noise) == 0:
				self.noise = self._get_noise_seq()
			self.count += 1
			return self.noise.popleft()
		else:
			raise StopIteration()


class noise15_iter:
	def __init__(self):
		self.seed = 1
		self.rand_gen = np.random.RandomState(self.seed)
		self.n = np.inf
		self.e = 0
		self.count = 0

	def __iter__(self):
		return self

	def __next__(self):
		if self.count == 0:
			self.e = self.rand_gen.randn()
		elif self.count < self.n:
			self.e = .7 * (self.e + self.rand_gen.randn())
		else:
			raise StopIteration()
		eps = johnson_transform_SU(-5.47,
								   15.9574,
								   -0.5444,
								   1.6898,
								   self.e)
		self.count += 1
		return eps



if __name__ == '__main__':
	main()
	
	
	


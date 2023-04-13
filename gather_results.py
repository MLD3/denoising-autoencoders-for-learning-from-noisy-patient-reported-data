import numpy as np
import os,sys
import matplotlib.pyplot as plt
import joblib
import scipy.stats



#NOTES:
#This code generates all results, once models have been run as when using the commands in CTDAE.py
#The first three sections generate:
#remaining MSE after denoising used in the table, 
#alternate noise plots, and
#CRC results for simulated and real data used in the table.

###############################IMPORTANT!!!!####################################
#The fourth section gathers time in range data. 
#It will only run if 'BB' is passed in as an argument.
#It requires this publically available implentation of the UVa/Padova simulator to be downloaded and available:
#https://github.com/jxx123/simglucose
#also, the simulator must be updated to use a second scenerio which passes noisy carbs to the bb controller. (my updated code is env.py and basal_bolus_ctrller.py in the simulated_data folder attached)
#It will take several hours to run.






#################################### gather simulated data MSE results########################

tsaves=[[] for i in range(10)]#for carrying over true carb values for SUP models
subs=range(10)
mods=['CAE','COTEACHP','NAE','N2N','SUP','SUPCT']
modnames=['CAE','CTDAE','NAC','N2N','SUP','SUPCT']
e=np.zeros((len(mods),10)) #stores remaining error
cis=np.zeros((len(mods),10,2)) #stores confidence intervals


for s in subs:
	mm=0
	for m in mods:
		#read in final results of model
		for o in os.listdir('SIMGLU-SUB'+str(s)+'-'+m):
			if o.endswith('f1'):
				fmse=float(o[:10])
		e[mm,s]=fmse

		#load in carb values
		x=joblib.load('SIMGLU-SUB'+str(s)+'-'+m+'/xs.pkl')[0][:,0,2]*200
		t=joblib.load('SIMGLU-SUB'+str(s)+'-'+m+'/xstru.pkl')[0][:,0,2]*200
		
		#noisy and true carbs are set to 0 for the supervised setting so need 
		#to pull values from a different model.
		if 'CAE' in m:
			tsaves[s]=t
		if 'SUP' in m:
			t=tsaves[s]
		
		c=joblib.load('SIMGLU-SUB'+str(s)+'-'+m+'/cs.pkl')[0]*200
		xo=x.copy()
		x=x+c
		x[x<0]=0
		mse=(x-t)**2
		if m=='N2N':
			#need to do the transform here
			xt=2*x-xo
			mse=mse=(xt-t)**2
		allerrs=[]
		#boot strap for confidence intervals
		for t in range(1000):
			inds=np.random.randint(x.shape[0],size=x.shape[0])
			allerrs.append(np.mean(mse[inds]))
		cis[mm,s,0]=np.percentile(allerrs,2.5)
		cis[mm,s,1]=np.percentile(allerrs,97.5)
		mm+=1

for mmm in range(len(mods)):
	print(mods[mmm]+' '+str(int(100*np.mean(e[mmm]))/100)+' ('+str(int(100*np.mean(cis[mmm,:,0]))/100)+','+str(int(100*np.mean(cis[mmm,:,1]))/100)+')')












#################################### Make alternate noise plot ########################


subs=range(10)
mods=['CAE','COTEACHP','NAE','N2N','SUP','SUPCT']
modnames=['CAE (Oracle)','N$^+$2N (ours)','NAC','NR2N','SUP','SUPCT']
noises=['ALT1','0','ALT4','ALT3','ALT7','ALT2','ALT6','ALT5']
noisenames=['x N 0','x N -','+ N 0','+ N -','x U 0','x U -','+ U 0','+ U -']
tsavest=[[] for i in range(10)] #for carrying over true carb values for SUP models
tsaves=[tsavest.copy() for n in noises] #for carrying over true carb values for SUP models
e=np.zeros((len(mods),len(noises),len(subs)))  #stores remaining error
stds=np.zeros((len(mods),len(noises),len(subs)))  #stores STDs for error bars
nn=0
for n in noises:
	for s in subs:
		mm=0
		for m in mods:
			directory='SIMGLU-SUB'+str(s)+'-VARS'+n+'-'+m
			if n=='0':
				directory='SIMGLU-SUB'+str(s)+'-'+m
			#read model output
			for o in os.listdir(directory):
				if o.endswith('f1'):
					fmse=float(o[:10])
			e[mm,nn,s]=fmse

			#load carb values
			x=joblib.load(directory+'/xs.pkl')[0][:,0,2]*200
			tt=joblib.load(directory+'/xstru.pkl')[0][:,0,2]*200
			
			#noisy and true carbs are set to 0 for the supervised setting so need 
			#to pull values from a different model.
			if 'CAE' in m:
				tsaves[nn][s]=tt
			if 'SUP' in m:
				tt=tsaves[nn][s]
			c=joblib.load(directory+'/cs.pkl')[0]*200
			xo=x.copy()
			x=x+c
			x[x<0]=0
			mse=(x-tt)**2
			if m=='N2N':
				#need to do the transform here
				xt=2*x-xo
				mse=(xt-tt)**2
			allerrs=[]
			#calculate error bars as stds from bootstraps
			for t in range(1000):
				inds=np.random.randint(x.shape[0],size=x.shape[0])
				allerrs.append(np.mean(mse[inds]))
			stds[mm,nn,s]=np.std(allerrs)
			mm+=1
	nn+=1

#generate figure.
plt.figure(figsize=(8,4))
wid=.8/float(len(mods))
for mmm in range(len(mods)):
	plt.bar(np.arange(len(noises))-.4+wid/2+wid*mmm,np.mean(e[mmm],1),width=wid)
for mmm in range(len(mods)):
	plt.errorbar(np.arange(len(noises))-.4+wid/2+wid*mmm,np.mean(e[mmm],1),yerr=np.mean(stds[mmm],1),capsize=wid,ls='none')
plt.ylabel('Remaining Carb MSE (10-Subject Average, $g^2$)')
plt.xticks(range(len(noises)),noisenames)
plt.xlabel('Noise Function')
plt.legend(modnames)
plt.savefig('NRnoise.png')
plt.clf()






#################################### gather real and simulated CRC results ########################

fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(16, 8))
#fig.tight_layout()
plt.rcParams['figure.constrained_layout.use'] = True
print('')
print('SIM CRC')
print('')

mm=0
m2=['CAE','N$^+$2N','NAC','NR2N','SUP','SUPCT']
for m in ['CAE','COTEACHP','NAE','N2N','SUP','SUPCT']:
	g=[]
	cs=[]
	for s in range(10):
		#load in glucose values (which are the same for all models)
		x=joblib.load('SIMGLU-SUB'+str(s)+'-CAE/xs.pkl')[0][:,12:,0]*400
		#load in updated carb values
		c=joblib.load('SIMGLU-SUB'+str(s)+'-'+m+'/cs.pkl')[0]*200
		if m=='N2N':
			#for noisier2noise, need to do the transform
			co=joblib.load('SIMGLU-SUB'+str(s)+'-CAE/xs.pkl')[0][:,0,2]*200
			c=(2*(c+co)-co)-co
		if 'SUP' in m:
			#for supervised, need to calculate difference from original value
			co=joblib.load('SIMGLU-SUB'+str(s)+'-CAE/xs.pkl')[0][:,0,2]*200
			c=c-co
		#calculate magni mean risk
		x=10*( 3.35506*(np.log(x)**.8353 - 3.7932))**2
		x=np.mean(x,1)
		#add risk scores and squared carb updates to list
		for xx in x:
			g.append(xx)
		for cc in c:
			cs.append(cc**2)
	g=np.array(g)
	c=np.array(cs)
	#report CRC
	print(m,scipy.stats.spearmanr(g,c))
	plt.subplot(2,6,mm+1)
	plt.title(m2[mm],fontSize=20)
	plt.plot(c,g,'o')
	m, b = np.polyfit(c, g, 1)
	plt.plot(c, m*c+b, 'k')
	if mm==0:
		plt.ylabel('Average Risk following Carb',fontSize=20)
	mm+=1

print('')
print('REAL CRC')
print('')
m2=['N$^+$2N','NAC','NR2N','SUP','SUPCT']
subs=['559','563','570','575','588','591','544','596']
mm=0
for m in ['COTEACHP','NAE','N2N','SUP','SUPCT']:
	g=[]
	cs=[]
	for s in range(8):
		#load in glucose values (which are the same for all models)
		x=joblib.load(str(subs[s])+'-NAE/xs.pkl')[0][:,12:,0]*400
		#load in updated carb values
		c=joblib.load(str(subs[s])+'-'+m+'/cs.pkl')[0]*200/5
		#NOTE! We divide by 5 here to match the per minute carb output of the simulator above.
		if m=='N2N':
			#for noisier2noise, need to do the transform
			co=joblib.load(str(subs[s])+'-NAE/xs.pkl')[0][:,0,2]*200/5
			c=(2*(c+co)-co)-co
		if 'SUP' in m:
			#for supervised, need to calculate difference from original value
			co=joblib.load(str(subs[s])+'-NAE/xs.pkl')[0][:,0,2]*200/5
			c=c-co
		#calculate magni mean risk
		x=10*( 3.35506*(np.log(x)**.8353 - 3.7932))**2
		x=np.mean(x,1)
		#add risk scores and squared carb updates to list
		for xx in x:
			g.append(xx)
		for cc in c:
			cs.append(cc**2)
	g=np.array(g)
	c=np.array(cs)
	#report CRC
	print(m,scipy.stats.spearmanr(g,c))
	plt.subplot(2,6,mm+8)
	#plt.title(m2[mm])
	plt.plot(c,g,'o')
	m, b = np.polyfit(c, g, 1)
	plt.plot(c, m*c+b, 'k')
	if mm==0:
		plt.xlabel('Magnitude of Carb correction ($g^2$)',fontSize=20)
	mm+=1
plt.savefig('CRC.png')
plt.clf()
###############################IMPORTANT!!!!####################################
#This section gathers time in range data. 
#It will only run if 'BB' is passed in as an argument.
#It requires this publically available implentation of the UVa/Padova simulator to be downloaded and available:
#https://github.com/jxx123/simglucose
#also, the simulator must be updated to use a second scenerio which passes noisy carbs to the bb controller. (my updated code is env.py and basal_bolus_ctrller.py in the simulated_data folder attached)
#It will take several hours to run.
if 'BB' in sys.argv:
	sys.path.append('/data3/interns/sim2real/simglucose')
	from simglucose.simulation.env import T1DSimEnv
	from simglucose.controller.basal_bolus_ctrller import BBController
	from simglucose.controller.base import Action
	from simglucose.sensor.cgm import CGMSensor
	from simglucose.actuator.pump import InsulinPump
	from simglucose.patient.t1dpatient import T1DPatient
	from simglucose.simulation.scenario_gen import RandomScenario
	from simglucose.simulation.scenario import CustomScenario
	from simglucose.simulation.sim_engine import SimObj, sim, batch_sim
	from datetime import timedelta
	from datetime import datetime
	from datetime import date

	#initialize start time.
	now = date.today()
	start_time = datetime.combine(now,datetime.min.time())

	SUBS=['01','02','03','04','05','06','07','08','09','10']
	mods=['clean','noisy','CAE','NAE','COTEACHP','N2N','SUP','SUPCT']
	outs=np.zeros((len(SUBS),len(mods),3))
	for m in range(len(mods)):
		mm=mods[m]
		ss=-1
		for s in SUBS:
			ss+=1
			#list of original carbs (oc) and cleaned carbs (cc) to gather
			cc=[]
			oc=[]
			#load in data and get test data set
			oo=joblib.load('/data3/interns/unc/adult'+s+'-0.pkl')
			oo=np.array(oo)[-50:]*5

			#track the number carb we are on
			cpos=0
			#load in carb updates but only if not doing just clean or noisy version
			if m>2:
				c=joblib.load('SIMGLU-SUB'+str(ss)+'-'+mm+'/cs.pkl')[0]*200*5
			#loop through data days
			for ooo in range(50):
				o=oo[ooo]
				#loop through timepoints
				for i in range(288-36):
					if o[i,4]!=0:
						oc.append([i/12.0+24.0*ooo,o[i,2]])#true carb value
						if m==0:
							cc.append([i/12.0+24.0*ooo,o[i,2]])#true carb value again
						elif m==1:
							cc.append([i/12.0+24.0*ooo,o[i,4]])#use noisy carb value
						elif m==6:
							#for NR2N, need to do transform
							temper=o[i,4]+c[cpos]
							temper=2*temper-o[i,4]
							cc.append([i/12.0+24.0*ooo,np.maximum(temper,0)])
						elif m>6:
							#for supervised method, the carb was learned from scratch.
							cc.append([i/12.0+24.0*ooo,np.maximum(c[cpos],0)])
						else:
							#for all other methods, update carb value.
							cc.append([i/12.0+24.0*ooo,np.maximum(o[i,4]+c[cpos],0)])
						cpos+=1 #increment carb count.

			#check to make sure carb updates match number of carbs.
			if m>2:
				if cpos!=len(c):
					print(cpos,len(c), s,mm)
					print('whoops.')
					quit()
				else:
					print('all good here.')
			
			#run simulation
			patient = T1DPatient.withName('adult#0'+s)
			sensor = CGMSensor.withName('GuardianRT',seed=0)
			pump = InsulinPump.withName('Insulet')
			scenario = CustomScenario(start_time=start_time, scenario=oc)
			scenario2 = CustomScenario(start_time=start_time, scenario=cc)
			env = T1DSimEnv(patient, sensor, pump, scenario,scenario2)
			controller = BBController(False)
			s1 = SimObj(env, controller,timedelta(days=20), animate=False, path='results')
			
			#gather results with time in range
			r = sim(s1)
			g=np.array(r['CGM'])#[-289:-1]
			ccc=np.array(r['CHO'])#[-289:-1]
			temp=g
			tir=float(len(temp[(temp>=70)*(temp<=180)]))/float(len(temp))

			#boot strap for CIs
			bs=[]
			for t in range(1000):
				gtemp=g[np.random.randint(g.shape[0],size=g.shape[0])]
				bs.append(float(len(gtemp[(gtemp>=70)*(gtemp<=180)]))/float(len(gtemp)))

			#save and dump output.
			outs[ss,m,0]=np.mean(tir)
			outs[ss,m,1]=np.percentile(bs,2.5)
			outs[ss,m,2]=np.percentile(bs,97.5)
			joblib.dump(outs,'BBcompout.pkl')
	for j in range(len(mods)):
		print(mods[j]+' '+str(int(10000*np.mean(outs[:,j,0]))/100)+' ('+str(int(10000*np.mean(outs[:,j,1]))/100)+','+str(int(10000*np.mean(outs[:,j,2]))/100)+')')

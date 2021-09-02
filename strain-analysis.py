import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
import MDAnalysis
import sys

#################################################
#             Input parameters                  #
#################################################

reference = str(sys.argv[1])
deformed = str(sys.argv[2])
out = str(sys.argv[3])
atom_selection = str(sys.argv[4])
cutoff = sys.argv[5]
stride_ref = sys.argv[6]
stride_def = sys.argv[7]

gro1 = reference + '.pdb'
xtc1 = reference + '.xtc'
gro2 = deformed + '.pdb'
xtc2 = deformed + '.xtc'
bar_fig = out +'_bar.svg'
strain_fig = out + '_strain.svg'
loaduser_file = out + '_strain_loaduser.dat'
strain_file = out + '_strain.dat'
sem_file = out + '_sem.dat'

##################################################
#             Loading trajectories               #
##################################################

traj1 = MDAnalysis.Universe(gro1, xtc1)
sele1 = traj1.select_atoms(atom_selection)
traj2 = MDAnalysis.Universe(gro2, xtc2)
sele2 = traj2.select_atoms(atom_selection)
if len(sele1) != len(sele2):
    raise Exception("The nr of atoms in both gro files don't match after truncation; reconsider your atom_selection argument")
resids = sele1.resids


##################################################
# Shear strain calculation (pairwise comparison) #
##################################################
R = cutoff
strain_dyn = []

strain_raw = np.zeros((sele1.atoms.n_atoms))
strain_sum = np.zeros((sele1.atoms.n_atoms))

step = 0
maxstep = len(traj1.trajectory[::stride_ref])*len(traj2.trajectory[::stride_def])
for ts1 in traj1.trajectory[::stride_ref]:
    x0 = sele1.positions
    dis_x0 = MDAnalysis.lib.distances.distance_array(x0, x0)
    for ts2 in traj2.trajectory[::stride_def]:
        step += 1
        x = sele2.positions
        tensor = []
        shear = []

        for i in range(len(x0)):
            B = []
            A = []


            for j in range(len(x0)):
                if dis_x0[i,j] < R and dis_x0[i,j] != 0.:
                    A.append(x0[j] - x0[i])
                    B.append(x[j] - x[i])
            Am = np.array(A)
            Bm = np.array(B)
            D = np.linalg.inv(Am.transpose()@Am)
            C = Bm@Bm.transpose() - Am@Am.transpose()
            Q = 0.5*(D@Am.transpose()@C@Am@D)

            tensor.append(Q)
            s = np.trace(Q@Q) - (1/3)*(np.trace(Q))**2
            shear.append(s)
            
        if step % int(maxstep/100) == 0:
            flog = open('strain.log','a+')
            flog.write('%d %% completed\n' % (step/int(maxstep/100)))
            flog.close()
        #shear = list(zip(shear, names, resids))
        #tensor =  list(zip(tensor, resids))
        #strain_dyn.append(shear)
        strain_raw = np.vstack((strain_raw,np.asarray(shear)))          
        
strain_avg = np.average(strain_raw[1:], axis=0)
sem = stats.sem(strain_raw[1:],axis=0)

###############################################################################

##############################################
#             Plotting graphs                #
##############################################


average = list(zip(strain_avg, resids)) 

def plot_loghist(x, bins):
  hist, bins = np.histogram(x, bins=bins)
  logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
  plt.hist(x, bins=logbins)
  plt.xscale('log')

plot_loghist(strain_avg,25)
plt.savefig(bar_fig, format='svg')


x = np.arange(resids[0],resids[-1]+1)

fig, ax1 = plt.subplots(figsize=(20, 10)) 
ax1.set_xlabel('Residue number',fontsize=18)
ax1.set_ylabel('Shear strain',fontsize=18)
ax1.plot(x,strain_avg,color='k')
ax1.set_xlim(left=1,right=resids[-1])
ax1.tick_params(axis="x", labelsize=14)
ax1.tick_params(axis="y", labelsize=14)
plt.locator_params(axis='x',nbins=50)
plt.xticks(rotation=45)
plt.savefig(strain_fig, format='svg')
plt.show()


#############################################
#         Write output to file              #
#############################################

fload = open(loaduser_file,'w')
for kk in range(sele1.atoms.n_atoms):
    for jj in range(len(average)):
        if sele1.atoms.resids[kk] == average[jj][1]:
            fload.write('%.16f ' % average[jj][0])
fload.close()

fstrain = open(strain_file,'w')
for kk in range(sele1.atoms.n_atoms):
    fstrain.write('%d %16f\n' % (x[kk], strain_avg[kk]))
fstrain.close()

fsem = open(sem_file,'w')
for kk in range(sele1.atoms.n_atoms):
    fsem.write('%.16f\n' % sem[kk])
fsem.close()




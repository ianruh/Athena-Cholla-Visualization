import h5py
from pylab import *  # imports numpy as np and matplotlib
import matplotlib.gridspec as gridspec
import athena_read   # Athena++'s reader
import array as ar   # for integer arrays
from PW15classes import *
from PW15classes import ti as blondin
from PW15classes import linearTheory
from PW15classes.linearTheory import getAnalyticSolution
from PW15classes.linearTheory import addAnalyticSolutions
import pickle
import multiprocessing as mp
import subprocess
import os
import sys
import warnings
import contextlib
from IPython import embed

if __name__ == '__main__':
	def printHelp():
		print("""
Usage: python animate [options]

Required:
	-c --cholla:	Directory containing the output files (*.h5)
			from a cholla run. Multiple directories can
			be listed, but another flag must be listed
			for every entry.

	Or

	-a --athena: 	Directory containing the ouput files from
			Athena++. Multiple directories can be included,
			but another flag must be listed for each entry.

Optional:
	-o --out: 	The output name for the mp4 movie. If one is
			not provided, then the animation will be shown
			in a gui instead.
	-l --label:	The label to put in the legend for the preceeding
			data set. If this is ommitted, then the name of the
			folder for the data set is used.
	
	*Note: 	If you are using IPython, you need to run it with:

			ipython script.py -- [arguments]

		The '--' tells IPython to pass the remaining options
		to the script.
		""")
	cholla_dirs = []
	athena_dirs = []
	cholla_labels = []
	athena_labels = []
	out_name = ""

	if(len(sys.argv) == 1):
		printHelp()
		sys.exit(1)
	try:
		i = 1
		while(i < len(sys.argv)):
			arg = sys.argv[i]
			if(arg == "-c" or arg == "--cholla"):
				cholla_dir = sys.argv[i+1]
				if(cholla_dir[-1] != '/'):
					cholla_dir += '/'
				cholla_dirs.append(cholla_dir)
				if(len(sys.argv) > i+2 and (sys.argv[i+2] == '-l' or sys.argv[i+2] == '--label')):
					cholla_labels.append(sys.argv[i+3])
					i+= 3
				else:
					cholla_labels.append([name for name in cholla_dir.split('/') if name != ''][-1])
					i += 1
				continue
			if(arg == "-a" or arg == "--athena"):
				athena_dir = sys.argv[i+1]
				if(athena_dir[-1] != '/'):
					athena_dir += '/'
				athena_dirs.append(athena_dir)
				if(len(sys.argv) > i+2 and (sys.argv[i+2] == '-l' or sys.argv[i+2] == '--label')):
					athena_labels.append(sys.argv[i+3])
					i += 3
				else:
					athena_labels.append([name for name in athena_dir.split('/') if name != ''][-1])
					i += 1
				continue
			if(arg == "-o" or arg == "--out"):
				out_name = sys.argv[i+1]
				if(out_name[-4:] != ".mp4"):
					out_name += ".mp4"
				i += 1
				continue
			if(arg == "-h" or arg == "--help"):
				printHelp()
				sys.exit(1)
			i += 1
	except:
		print("Unknown usage. Run with `--help` for help.".format(sys.argv[0]))
		sys.exit(1)

	if((len(cholla_dirs) == 0 and len(athena_dirs) == 0)):
		print("Unknown usage. Run with `--help` for help.".format(sys.argv[0]))
		sys.exit(1)
	
# embed()
###### For removing annoying stdout
class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
	save_stdout = sys.stdout
	save_stderr = sys.stderr
	sys.stdout = DummyFile()
	sys.stderr = sys.stdout
	yield
	sys.stdout = save_stdout
	sys.stderr = save_stderr

###### End hackish getaround

""" Inputs """	
if __name__ == '__main__':
	MAKE_MOVIE = out_name != ""
	plot_analytic = 1
	plot_athena_solution = len(athena_dirs) != 0
	if(len(cholla_dirs) > 0):
		i_dumps = ar.array('i',(i for i in range(0,len(os.listdir(cholla_dirs[0]))-1,1)))
	else:
		i_dumps = ar.array('i',(i for i in range(0,len(os.listdir(athena_dirs[0]))-1,1)))
	Ndumps = len(i_dumps)

ADD_ALL_NODES = 0
JUST_SOUND_WAVES = 0
RESIDUALS = 0
TC = 1
DE = 0
BLONDIN = 1
STATIC_AXES = True

nT = 1e13
xi_b = 190.
xf = 1.
dt = 0.25 # time between dumps in code units
dump_tag = 'ti.block0.out1.'

Amp = 1e-4 # amplitude of perturbation

tempDir = '.tmp/'
if __name__ == '__main__' and MAKE_MOVIE:
	subprocess.call(['mkdir', '.tmp'])

# padding for dens-vel-pres panels
pad_d = 100
pad_v = 10
pad_p = 1e4


""" Inputs for dispersion relation calc """

gamma = 5./3.

# calculate H/C derivatives
if BLONDIN:
	hcModel = blondin.HCcurves(gamma,nT=nT)
	eqvals = hcModel.getEquilibriumValues(xi_b)
	athena = hcModel.getCodeUnits(eqvals,1.)
	L_T,L_rho = hcModel.getL_TandL_rho(xi_b,eqvals)

	# Conduction
	field_length = hcModel.getFieldLength(eqvals)/athena['lambda']
else:
	L_T,L_rho = 0.5,1.

	# Conduction
	field_length = 0.2

# TI parameters
Mach0 = 0.
cs0 = 1.
K1 = 1.
IVs = {}
IVs['density'] = 1.
IVs['pressure'] = 1./gamma
IVs['sound speed'] = cs0
IVs['velocity'] = IVs['sound speed']*Mach0
IVs['A'] = Amp
IVs['M0'] = Mach0

norms = {}
norms['density'] = IVs['density']
norms['velocity'] = cs0
norms['pressure'] = IVs['density']*cs0**2

#wave properties
xo = 0.
xc = 0.5*xf 
Nwaves  = 1.
wavelength = (xf - xo)/Nwaves
k = 2.*np.pi/wavelength #wavenumber

# basic TI dispersion relation parameters (M0 allows for nonzero initial velocity)
TIparams = {'gamma':gamma, 'cs0':cs0, 'k':k, 'tau_th':K1, 'L_T':L_T, 'L_rho':L_rho, \
           'lambdaF':field_length, 'K':1., 'M0':Mach0}

# RADFORCE PARAMETERS - dispersion relation accounting for rad force			
TIparams['F_T'] = 0.
TIparams['F_S'] = 0.
RFparams = {'alpha':0., 'beta':0., 'tau_rad':1.}

# get the coefficients needed in analytic solution
coeffs = linearTheory.getCoeffs(TIparams,RFparams,IVs['pressure'],TC=TC)


""" Load Athena grid data """	

# # load cell edge grid
# if(plot_athena_solution):
# 	dump = dump_tag + str(0).zfill(5) + '.tab'
# 	athena_dic = athena_read.tab(dump_dir_athena + dump, ['x', 'rho', 'pgas', 'v1', 'v2', 'v3'])
# 	x = athena_dic['x'][0,0,:]
# 	d = athena_dic['rho'][0,0,:]
# 	p = athena_dic['pgas'][0,0,:]
# 	v = athena_dic['v1'][0,0,:]

""" Load or calculate equilibrium curve """
xis = np.logspace(1.,4.,150)
Ts = np.logspace(4.,7.,150)
X,Y = np.meshgrid(xis,Ts)
try:
	L = pickle.load(open(os.path.dirname(__file__) + '/pickled/Blondin_eqcurve.p', "rb"))
	Z = pickle.load(open(os.path.dirname(__file__) + '/pickled/Blondin_balbuscontour.p', "rb"))
except:
	Z = hcModel.sampleBalbusCriterion(xis,Ts)
	L = hcModel.sampleNetLossFunction(xis,Ts)
	if __name__ == '__main__':
		pickle.dump(L, open(os.path.dirname(__file__) + '/pickled/Blondin_eqcurve.p', "wb"), protocol=2)
		pickle.dump(Z, open(os.path.dirname(__file__) + '/pickled/Blondin_balbuscontour.p', "wb"), protocol=2)
	
""" Plotting routine """	
colors = ['k','b','r']
linestyles = ['-','--','--']
linewidths=[2,2,1]

if __name__ == '__main__' and STATIC_AXES:
	def max(num1, num2):
		if(num1 > num2):
			return num1
		return num2

	def min(num1, num2):
		if(num1 < num2):
			return num1
		return num2

	maxRho = -1e10
	minRho = 1e10
	maxV = -1e10
	minV = 1e10
	maxT = -1e10
	minT = 1e10
	for i in range(0,Ndumps, 1):
		for dump_dir in cholla_dirs:
			
			f = h5py.File(dump_dir+str(i_dumps[i])+'.h5.0', 'r')
			head = f.attrs
			nx = head['dims'][0]
			x = np.array(range(0,nx,1))
			d  = np.array(f['density']) # mass density
			mx = np.array(f['momentum_x']) # x-momentum
			my = np.array(f['momentum_y']) # y-momentum
			mz = np.array(f['momentum_z']) # z-momentum
			E  = np.array(f['Energy']) # total energy density
			v = mx/d
			if DE:
				e  = np.array(f['GasEnergy'])
				p  = e*(gamma-1.0)
				ge = e/d
			else:
				p  = (E - 0.5*d*v**2) * (gamma - 1.0)
				ge  = p/d/(gamma - 1.0)

			t = eqvals['T0']*gamma*p/d
			maxRho = max(maxRho, d.max())
			minRho = min(minRho, d.min())
			maxV = max(maxV, v.max())
			minV = min(minV, v.min())
			maxT = max(maxT, t.max())
			minT = min(minT, t.min())
		    		
def plotter(bundle):
	global x
	i = bundle["i"]
	MAKE_MOVIE = bundle["make_movie"]
	plot_analytic = bundle["plot_analytic"]
	plot_athena_solution = bundle["plot_athena_solution"]
	cholla_dirs = bundle["cholla_dirs"]
	labels = bundle["cholla_labels"]
	athena_dirs = bundle["athena_dirs"]
	athena_labels = bundle["athena_labels"]
	i_dumps = bundle["i_dumps"]
	ax1 = bundle["ax1"]
	ax2 = bundle["ax2"]
	ax3 = bundle["ax3"]
	ax4 = bundle["ax4"]
	ax5 = bundle["ax5"]
	if(STATIC_AXES):
		maxRho = bundle["maxRho"]
		minRho = bundle["minRho"]
		maxV = bundle["maxV"]
		minV = bundle["minV"]
		maxT = bundle["maxT"]
		minT = bundle["minT"]
	
	if(plot_athena_solution):
		run_name_athena = athena_labels[0]
		dump_dir_athena = athena_dirs[0]
	
	if(not MAKE_MOVIE):
		ax1.cla()
		ax2.cla()
		ax3.cla()
		ax5.cla()
		Nlines = len(ax4.lines)
		if Nlines > 0: 
			for n in range(Nlines):
				del ax4.lines[0]

	# set limits
	if RESIDUALS:
		plt.axis([0., x[-1], -pad_d*Amp, pad_d*Amp])
		plt.axis([0., x[-1], -pad_v*Amp, pad_v*Amp])
		plt.axis([0., x[-1], -(pad_p*Amp)**2, pad_p*Amp])

	ax1.set_title('PW15 TI test')

	#Display time on window
	time_template = 'time = %.1f'
	time_text = ax1.text(0.75, 0.9, '', transform=ax1.transAxes) 
	
	# update time
	time = dt*float(i_dumps[i]) # time in code units
	time_text.set_text(time_template%(time))
	
	for dump_dir,color,ls,lw,label in zip(cholla_dirs,colors,linestyles,linewidths,labels):
		
		f = h5py.File(dump_dir+str(i_dumps[i])+'.h5.0', 'r')
		head = f.attrs
		nx = head['dims'][0]
		# gamma = head['gamma'][0]
		# x = np.arange(0.5*head['dx'][0],1.,head['dx'][0])
		x = np.array(range(0,nx,1))
		d  = np.array(f['density']) # mass density
		mx = np.array(f['momentum_x']) # x-momentum
		my = np.array(f['momentum_y']) # y-momentum
		mz = np.array(f['momentum_z']) # z-momentum
		E  = np.array(f['Energy']) # total energy density
		v = mx/d
		if DE:
		  e  = np.array(f['GasEnergy'])
		  p  = e*(gamma-1.0)
		  ge = e/d
		else:
		  p  = (E - 0.5*d*v**2) * (gamma - 1.0)
		  ge  = p/d/(gamma - 1.0)
		  
		if RESIDUALS:
			d -= IVs['density']
			p -= IVs['pressure']
		cs2 = gamma*p/d
		cs = np.sqrt(cs2)
		M = v/cs
		t = eqvals['T0']*gamma*p/d
		logT_cgs = np.log10(eqvals['T0']*gamma*p/d)
		logxi_cgs = np.log10(xi_b/d)
	# 	B = 0.5*v**2 + cs2/(gamma - 1.) 
	
		# check to see if floors are hit
		# d_min,d_max = np.min(d),np.max(d)
		# p_min = np.min(p)
		# print("time = {}, d_min = {}, d_max = {}, p_min = {}, p_max = {}".format(time,d_min,d_max,p_min,np.max(p)))
	
		# plot Cholla solution
		ax1.plot(x/270,d,ls=ls,lw=lw,color=color,label=label)
		ax2.plot(x/270,v,ls=ls,lw=lw,color=color)
		ax3.plot(x/270,p,ls=ls,lw=lw,color=color)
		ax5.plot(x/270,t,ls=ls,lw=lw,color=color)

	if plot_athena_solution: #load/unpack Athena data
			for directory,label in zip(athena_dirs, athena_labels):
				dump = directory + dump_tag + str(i_dumps[i]).zfill(5) + '.tab'
				athena_dic = athena_read.tab(dump, ['x', 'rho', 'pgas', 'v1', 'v2', 'v3'])
				x_ath = athena_dic['x'][0,0,:]
				d_ath = athena_dic['rho'][0,0,:]
				p_ath = athena_dic['pgas'][0,0,:]
				v_ath = athena_dic['v1'][0,0,:]
				s_ath = np.log(p_ath/d_ath**gamma)
				
				if RESIDUALS:
					d_ath -= IVs['density']
					p_ath -= IVs['pressure']
				
				ax1.plot(x_ath,d_ath,ls='-',lw=1,color='orange',label=label)
				ax2.plot(x_ath,v_ath,ls='-',lw=1,color='orange')
				ax3.plot(x_ath,p_ath,ls='-',lw=1,color='orange')
				ax5.plot(x_ath,s_ath,ls='-',lw=1,color='orange')
				
				# plot tracks
				logT_cgs_ath = np.log10(eqvals['T0']*gamma*p_ath/d_ath)
				logxi_cgs_ath = np.log10(xi_b/d_ath)
				ax4.plot(logxi_cgs_ath,logT_cgs_ath,ls='',marker='o',color='orange')
			
	if plot_analytic: # plot analytic solution
		global d_analytic,v_analytic,p_analytic
		cmode = 1
		d_cmode = getAnalyticSolution('density',x,time,coeffs,IVs,norms,Nwaves=Nwaves,resid=RESIDUALS)
		v_cmode = getAnalyticSolution('velocity',x,time,coeffs,IVs,norms,Nwaves=Nwaves,resid=RESIDUALS)
		p_cmode = getAnalyticSolution('pressure',x,time,coeffs,IVs,norms,Nwaves=Nwaves,resid=RESIDUALS)
		
		if ADD_ALL_NODES or JUST_SOUND_WAVES:
			if JUST_SOUND_WAVES:
				cmode = 0
			d_analytic = addAnalyticSolutions('density',x,time,TIparams,IVs,norms,conduction=TC,Nwaves=Nwaves,resid=RESIDUALS,cmode=cmode)
			v_analytic = addAnalyticSolutions('velocity',x,time,TIparams,IVs,norms,conduction=TC,Nwaves=Nwaves,resid=RESIDUALS,cmode=cmode)
			p_analytic = addAnalyticSolutions('pressure',x,time,TIparams,IVs,norms,conduction=TC,Nwaves=Nwaves,resid=RESIDUALS,cmode=cmode)	
			if JUST_SOUND_WAVES:
				d_analytic += IVs['density']
				v_analytic += IVs['velocity']
				p_analytic += IVs['pressure']		
		else:
			d_analytic = d_cmode
			v_analytic = v_cmode
			p_analytic = p_cmode
		with nostdout():
			s_analytic = np.log(p_analytic/d_analytic**gamma)
		
		ax1.plot(x,d_analytic,'gray',lw=2,ls='--',label='Analytic')
		ax2.plot(x,v_analytic,'gray',lw=2,ls='--')
		ax3.plot(x,p_analytic,'gray',lw=2,ls='--')
		ax5.plot(x,s_analytic,'gray',lw=2,ls='--')
	
	
	ax1.legend(loc='upper left')
	
	# plot tracks of last sim 
	ax4.plot(logxi_cgs,logT_cgs,ls='',marker='.',color=color)
	if(STATIC_AXES):
		ax1.set_ylim([minRho, maxRho])
		ax2.set_ylim([minV, maxV])
		ax5.set_ylim([minT, maxT])
	if(MAKE_MOVIE):
		plt.savefig(tempDir + str(i) + '.png', dpi=200)
		plt.close()

""" Code to save image, animate, or make movie """
if __name__ == '__main__':
	fig = figure(figsize=(11.5, 7))

	""" Figure setup plus a hack function """	

	#compatible with MNRAS style file when saving figs
	# matplotlib.rcParams['mathtext.fontset'] = 'stix'
	# matplotlib.rcParams['font.family'] = 'STIXGeneral'
	# matplotlib.rcParams['font.size'] = '16'
	# matplotlib.rcParams['ps.fonttype'] = 42 

	fs = 12

	spec3 = gridspec.GridSpec(ncols=2, nrows=3)

	ax1 = fig.add_subplot(spec3[0, 0])
	ylabel(r'$\rho$',fontsize=fs)

	ax2 = fig.add_subplot(spec3[1, 0])
	ylabel(r'$v$',fontsize=fs)

	ax3 = fig.add_subplot(spec3[2, 0])
	ylabel(r'$p$',fontsize=fs)
	xlabel(r'$x$',fontsize=fs)

	ax4 = fig.add_subplot(spec3[0:2, 1:])
	ylabel(r'$\log(T)$',fontsize=fs)
	xlabel(r'$\log(\xi)$',fontsize=fs)
	
	# eq_line = ax4.lines[0]
	# balbus_line = ax4.lines[1]
	eq_contour = plt.contour(np.log10(X),np.log10(Y),L,[0.])
	eq = eq_contour.collections[0]
	plt.setp(eq, linewidth=2, color ='k')

	balbus_contour = plt.contour(np.log10(X),np.log10(Y),Z,[0.])
	balbus = balbus_contour.collections[0]	
	plt.setp(balbus, linewidth=2, color ='gray', linestyle='--')

	ax5 = fig.add_subplot(spec3[2, 1:])
	ylabel(r'$T$',fontsize=fs)
	xlabel(r'$x$',fontsize=fs)

	plt.subplots_adjust(left=0.11, bottom=0.09, right=0.97, top=0.94, wspace=0.26, hspace=0.3)	

	def bundler(i):
		info = {
			"i": i,
			"make_movie": MAKE_MOVIE,
			"plot_athena_solution": plot_athena_solution,
			"plot_analytic": plot_analytic,
			"cholla_dirs": cholla_dirs,
			"cholla_labels": cholla_labels,
			"athena_dirs": athena_dirs,
			"athena_labels": athena_labels,
			"i_dumps": i_dumps,
			"ax1": ax1,
			"ax2": ax2,
			"ax3": ax3,
			"ax4": ax4,
			"ax5": ax5
		}
		if(STATIC_AXES):
			info.update({
				"maxRho": maxRho,
				"minRho": minRho,
				"maxV": maxV,
				"minV": minV,
				"maxT": maxT,
				"minT": minT,
			})
		return info

	mp.set_start_method('forkserver')
	if MAKE_MOVIE:
		pool = mp.Pool(None)
		
		# The easiest way to pass more arguments to the plotter function is 
		# by bundling it with each index.
		bundle = map(bundler, range(0, Ndumps, 1))
		for i, _ in enumerate(pool.imap(plotter, bundle, chunksize=1), 1):
			perc = (int(i/Ndumps*100))
			print('    [\u001b[32;1m' + ('#'*perc) + (' '*(100-perc)) + '\u001b[0m] {}%    '.format(perc), end="\r")
		print("")
		subprocess.call(['ffmpeg', '-hide_banner', '-r', '10', '-s', '1800x1800', '-i', tempDir+'%d.png', '-crf', '25', '-pix_fmt', 'yuv420p', out_name])
		subprocess.call(['rm', '-r', '.tmp'])
	else:
		import matplotlib.animation as animation

		def plotWrapper(i):
			bundle = bundler(i)
			plotter(bundle)
		
		ani = animation.FuncAnimation(fig, plotWrapper, frames=Ndumps, interval=200, blit=False, repeat=False)
		show()

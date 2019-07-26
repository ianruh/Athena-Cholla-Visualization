import sys
import numpy as np
from scipy.optimize import brentq,newton,bisect

def getCoeffs(TIparams,RFparams,P0,FIELD=1,TC=0,EQ=1,DEBUG=False):
	DR = TImodes(TIparams,Conduction=TC,Equilibrium=EQ)
	use_rootfinder = False
	#step 1: solve for density coefficients
	if FIELD:
		if use_rootfinder:
			n = DR.rootFinder('Field')
			nR = n.real
			nI = n.imag
			if 1==0:
				print("Field's root = {}".format(nR))
		else:
			roots,errors = DR.getRoots('Field')
			if 1==0:
				print("Roots =\n",roots)
				print("Errors = \n",errors)
			Nroots = len(roots)
			nRs = np.zeros(Nroots)
			nIs = np.zeros(Nroots)
			for r in range(Nroots):
				nRs[r] = roots[r].real
				nIs[r] = roots[r].imag
			rmax = nRs.argmax() #maximum real growth rate
			nR = nRs[rmax]
			nI = nIs[rmax]
	else:
		roots,errors = DR.getRoots('PWR',RFparams)
		if DEBUG:
			print("Roots =\n",roots)
			print("Errors = \n",errors)
		Nroots = len(roots)
		nRs = np.zeros(Nroots)
		nIs = np.zeros(Nroots)
		for r in range(Nroots):
			nRs[r] = roots[r].real
			nIs[r] = roots[r].imag
		rmax = nRs.argmax() #maximum real growth rate
		nR = nRs[rmax]
		nI = nIs[rmax]
	#step 2: solve for corresponding pressure coefficients
	Av_r,Av_i = DR.velocityAmplitudes(nR,nI)
	Ap_r,Ap_i = DR.pressureAmplitudes(P0,nR,nI)
	return {'nR':nR,'nI':nI,'Av_r':Av_r,'Av_i':Av_i,'Ap_r':Ap_r,'Ap_i':Ap_i}

def get_all_coeffs(TIparams,P0,TC=0):
	DR = TImodes(TIparams,Conduction=TC)
	roots,errors = DR.getRoots('Field')
	Nroots = len(roots)
	
	all_coeffs = [{},{},{}]
	for n in range(Nroots):

		nR,nI = roots[n].real,roots[n].imag
		Av_r,Av_i = DR.velocityAmplitudes(nR,nI)
		Ap_r,Ap_i = DR.pressureAmplitudes(P0,nR,nI)
		all_coeffs[n] = {'nR':nR,'nI':nI,'Av_r':Av_r,'Av_i':Av_i,'Ap_r':Ap_r,'Ap_i':Ap_i}
# 		all_coeffs[n] = {'nR':nR,'nI':nI,'Ap_r':AR,'Ap_i':AI}
		print("[get_all_coeffs]: n = {}, (nR,nI) = ({},{})".format(n,all_coeffs[n]['nR'],all_coeffs[n]['nI']))
	
	return all_coeffs
	
def getAnalyticSolution_old(var,x_array,time,coeffs,IVs,wavelength=1,HC=True,amp=None,residual=False):
	if not(HC):
		time = 0.
	nR = coeffs['nR']
	nI = coeffs['nI']
	if amp == None:
		A = IVs['A']
	else:
		A = amp
		
	# compute waveforms up front
	k = 2.*np.pi/wavelength
	xc = 0.5*wavelength
	coskx = np.cos(k*(x_array-xc))
	sinkx = np.sin(k*(x_array-xc))
	
	# apply filter - useful for a finite wave
# 	filter = [1 if x>=xo and x<=xf else 0. for x in x_array]
# 	coskx = filter*coskx
# 	sinkx = filter*sinkx 
	
	if var == 'density':
		A0 = IVs['density']
		A1 = 1.
		AR = A
		AI = 0.
	elif var == 'velocity':
		A0 = IVs['sound speed']
		A1 = IVs['M0']
		AR = -A*(nI/k + A1)
		AI = A*nR/k
	elif var == 'pressure':
		A0 = IVs['pressure']
		A1 = 1.
		AR = A*coeffs['Ap_r']
		AI = A*coeffs['Ap_i']
	R = np.exp(nR*time)*(np.cos(nI*time)*(AR*coskx - AI*sinkx) - np.sin(nI*time)*(AI*coskx + AR*sinkx))
	if residual:
		return R
	else:
		return A0*(A1 + R)

def getAnalyticSolution(var,z_axis,time,coeffs,IVs,norms,Nwaves=1,resid=0,cosine_wave=1):
	# determine wavenumber, k
	Ltot = np.max(z_axis) # - np.min(z_axis)
	wavelength = Ltot/Nwaves
	k = 2.*np.pi/wavelength

	nR = coeffs['nR']
	nI = coeffs['nI']
	A = IVs['A']
	if var == 'density':
		AR = A
		AI = 0.
	elif var == 'velocity':
		AR = A*coeffs['Av_r']
		AI = A*coeffs['Av_i']
	elif var == 'pressure':
		AR = A*coeffs['Ap_r']
		AI = A*coeffs['Ap_i']
	
	# q is the background flow, delta_q the Eulerian perturbation
	xc = 0.5*wavelength
	coskz = np.cos(k*(z_axis-xc))
	sinkz = np.sin(k*(z_axis-xc))
# 	coskz = np.cos(k*z_axis)
# 	sinkz = np.sin(k*z_axis)
	cos_nI = np.cos(nI*time)
	sin_nI = np.sin(nI*time)
	q = IVs[var]
	if cosine_wave:
		delta_q = norms[var]*np.exp(nR*time)*(coskz*(AR*cos_nI - AI*sin_nI) - sinkz*(AR*sin_nI + AI*cos_nI))
	else:
		delta_q = norms[var]*np.exp(nR*time)*(sinkz*(AR*cos_nI - AI*sin_nI) + coskz*(AR*sin_nI + AI*cos_nI))
	
	if resid:
		return delta_q
	else:
		return q + delta_q
		
def addAnalyticSolutions(var,z_axis,time,TIparams,IVs,norms,conduction=0,Nwaves=1,resid=0,cosine_wave=1,cmode=1):
	all_coeffs = get_all_coeffs(TIparams,conduction)
	
	nRs = np.zeros(3)
	for n in range(3):
		nRs[n] = all_coeffs[n]['nR']

	n_pos = nRs.argmax() #maximum real growth rate
	if n_pos == 0:
		sound_wave_coeffs = [all_coeffs[1],all_coeffs[2]]
	elif n_pos == 1:
		sound_wave_coeffs = [all_coeffs[0],all_coeffs[2]]
	else:
		sound_wave_coeffs = [all_coeffs[0],all_coeffs[1]]
	
	soln = np.zeros_like(z_axis)
	if cmode:
		soln += getAnalyticSolution(var,z_axis,time,all_coeffs[n_pos],IVs,norms,Nwaves=Nwaves,resid=resid,cosine_wave=cosine_wave)
	
	# use resid = True for remaining two modes
	for coeffs in sound_wave_coeffs:
		soln += getAnalyticSolution(var,z_axis,time,coeffs,IVs,norms,Nwaves=Nwaves,resid=1,cosine_wave=cosine_wave)
	return soln
	
def getKappa(n,T,pow): 
	CoulombLog = 20. #29.7 + np.log((T/1e6)/np.sqrt(n))
	psi = 1.84e-5/CoulombLog
	return psi*T**pow
	
def getFieldLength(eqvals,pow=2.5):
	T = eqvals['T0']
	rho = eqvals['d0']
	Omega = eqvals['Omega0']
	kappa = getKappa(eqvals['n0'],T,pow)
	return 2.*np.pi*np.sqrt(kappa*T/(Omega*rho))
		
class TImodes:
	def __init__(self,params,Conduction=False,Equilibrium=True,DEBUG=False):
		self.tau_th = 1. #params['tau_th']
		self.g = params['gamma']
		self.k = params['k']
		self.K = params['K']
		self.M0 = params['M0']
		self.cs0 = params['cs0']
		#self.tau_sc = (2.*np.pi/self.k)/self.cs
		self.L_T = params['L_T']
		self.L_rho = params['L_rho']
		self.F_T = params['F_T']
		self.F_S = params['F_S']
		if Conduction:
			#self.L_T += params['lambdaF']**2.
			self.L_T += (self.k*params['lambdaF']/(2.*np.pi))**2.
		if not(Equilibrium):
			self.L_rho += params['netLosses']
		self.L_S = self.L_T - self.L_rho
		self.n0 = 1j*self.k*self.M0
		self.DEBUG = DEBUG
	def prepareCubic(self):
		self.R = (2.*self.a**3 + -9.*self.a*self.b + 27.*self.c)/54.
		self.Q = (self.a**2 - 3.*self.b)/9.
		if self.DEBUG:
			print("R={}".format(self.R))
			print("Q={}".format(self.Q))
		RisReal = False
		QisReal = False
		if self.R == np.conjugate(self.R):
			RisReal = True
		if self.Q == np.conjugate(self.Q):
			QisReal = True
		if RisReal and QisReal and self.R.real**2-self.Q.real**3 < 0.:
			self.allRealRoots = True
			self.T = np.arccos(self.R/self.Q**1.5)
			if self.DEBUG:
				print("T = {}".format(self.T))
		else:
			self.allRealRoots = False
			discrim = np.sqrt(self.R**2-self.Q**3)
			D = complex(discrim.real,discrim.imag)
			R = complex(self.R.real,self.R.imag)
			test = R.real*D.real + R.imag*D.imag # same as Re(conjugate(R)*D)
			if self.DEBUG:
				print("discrim = {}, test = {}".format(discrim,test))
			if test < 0.:
				D *= -1.
			self.A = -(R + D)**(1./3.)
			if self.A == complex(0.,0.):
				self.B = complex(0.,0.)
			else:
				self.B = self.Q/self.A
		if self.DEBUG:
			print("(A,B)=")
			print(self.A,self.B)
	def setCubicField(self):
		if self.DEBUG:
			print("======================")
			print("    Field's cubic     ")
			print("======================")
		#print "K = {}".format(self.K)
		self.a = self.L_T/self.K
		self.b = self.k**2 + 1j*self.k*(self.F_S - self.g*self.F_T)
		self.c = self.L_S*self.k**2/(self.g*self.K) \
				+ 1j*(self.k/self.K)*(self.L_S*self.F_T - self.F_S*self.L_T)
		#print "a = {}, b = {}, c = {}".format(self.a,self.b,self.c)
		self.prepareCubic()
	def setCubicPWR(self,params):
		if self.DEBUG:
			print("======================")
			print("      PWR's cubic     ")
			print("======================")
		alpha = params['alpha']
		beta = params['beta']
		tau = params['tau_rad']
		kcs = self.k*self.cs0
		denom = (tau*kcs/self.g)**2. - beta**2.
		Zreal = (alpha-beta)*beta/denom
		Zimag = (beta-alpha)*(kcs/self.g)*tau/denom
		self.Z = complex(Zreal,Zimag)
		Wreal = (tau**2/self.g)/denom
		Wimag = (beta*tau/kcs)/denom
		self.W = complex(Wreal,Wimag)
		self.a = self.L_T/self.tau_th
		self.b = (self.g - self.Z)/self.W
		self.c = ((1.-self.Z)*self.L_T - self.L_rho)/(self.tau_th*self.W)
		self.prepareCubic()
	def velocityAmplitudes(self,nR,nI):
		Av_r = -nI/self.k
		Av_i = nR/self.k
		return Av_r,Av_i
	def pressureAmplitudes(self,p0,nR,nI):
# 		Preal = self.g*((nI**2 - nR**2)/self.k**2 + 2.*nI*self.M0/self.k + self.M0**2)
# 		Pimag = -2.*self.g*nR*(nI/self.k + self.M0)/self.k
		Preal = (nI**2 - nR**2)/self.k**2 + 2.*nI*self.M0/self.k + self.M0**2
		Pimag = -2.*nR*(nI/self.k + self.M0)/self.k
		return Preal,Pimag
	def cubicFunc(self,n):
		x = n+self.n0
		return x**3 + self.a*x**2 + self.b*x + self.c
	def setDispersion(self,choice,params=None):
		if choice == "Field":
			self.setCubicField()
		elif choice == "PWR":
			self.setCubicPWR(params)
		#Numerically solve for the thermal mode using Brent's method
		#return brentq(self.cubicFunc,1e-5,1e5,rtol=1e-28,maxiter=200,full_output=True,disp=True)
		root = newton(self.cubicFunc, 0.5, fprime=None, args=(), tol=1e-13, maxiter=50,fprime2=None)
		if self.DEBUG:
			print("Newton's method found root = {}".format(root))
		return root
	def getRoots(self,choice,params=None):
		if choice == "Field":
			self.setCubicField()
		elif choice == "PWR":
			self.setCubicPWR(params)
		if not(self.allRealRoots): #then either 1 real, 2 complex roots or all complex roots
			i = complex(0.,1.)
			x1 = self.A + self.B - self.a/3.
			x2 = -0.5*(self.A+self.B) - self.a/3. - i*0.5*np.sqrt(3.)*(self.A-self.B)
			x3 = -0.5*(self.A+self.B) - self.a/3. + i*0.5*np.sqrt(3.)*(self.A-self.B)
		else: #three real roots
			x1 = -2.*np.sqrt(self.Q)*np.cos(self.T/3.) - self.a/3.
			x2 = -2.*np.sqrt(self.Q)*np.cos((self.T + 2.*np.pi)/3.) - self.a/3.
			x3 = -2.*np.sqrt(self.Q)*np.cos((self.T - 2.*np.pi)/3.) - self.a/3.
		roots = []; errors = []
		root1 = x1-self.n0; roots.append(root1); errors.append(self.cubicFunc(root1))
		root2 = x2-self.n0; roots.append(root2); errors.append(self.cubicFunc(root2))
		root3 = x3-self.n0; roots.append(root3); errors.append(self.cubicFunc(root3))
		return roots,errors
	
	def rootFinder(self,choice,params=None):
		if choice == "Field":
			self.setCubicField()
		elif choice == "PWR":
			self.setCubicPWR(params)
		#Numerically solve for the thermal mode using Brent's method
		#return brentq(self.cubicFunc,1e-5,1e5,rtol=1e-28,maxiter=200,full_output=True,disp=True)
		root = newton(self.cubicFunc, 0.5, fprime=None, args=(), tol=1e-13, maxiter=50,fprime2=None)
		if self.DEBUG:
			print("Newton's method found root = {}".format(root))
		return root
	def thermalTime(self,g,n):
		tau = (1. - 0.5*(1. + g*n**2/self.b))/(g*n*(1. + n**2/self.b))
		return tau
		

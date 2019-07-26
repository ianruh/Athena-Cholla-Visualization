import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import brentq,newton,bisect

class BBrootfinder:
	def __init__(self,func,minRange=1e-2,maxRange=1e2,Nbrackets=200,tolerance=1e-10):
		self.func = func
		self.MAX_ROOTS = 4
		self.root_count = 0
		self.iter = 0
		self.TOL = tolerance
		self.xL = []
		self.xR = []
		self.roots = []
		self.minRange = minRange
		self.maxRange = maxRange
		self.Nbrackets = Nbrackets
	
	def bracketRoots(self):
		x1 = self.minRange
		x2 = self.maxRange
		dx = (x2-x1)/self.Nbrackets
		fprobe = self.func(x1)
		xstep = x1
		for i in range(self.Nbrackets):
			xstep += dx
			#print "dx = {}, xstep = {}".format(dx,xstep)
			fstep = self.func(xstep)
			fprod = fprobe*fstep
			if fprod < 0.:
				#print "xL = {}, xR = {}".format(xstep - dx,xstep)
				self.xL.append(xstep - dx)
				self.xR.append(xstep)
				self.root_count += 1
				if self.root_count == self.MAX_ROOTS: break
			fprobe = fstep
	
	def bisection(self):
		root = 0.
		for i in range(self.root_count):
			x1 = self.xL[i]
			x2 = self.xR[i]
			dx = x2-x1
			while np.abs(dx) > self.TOL:
				self.iter += 1
				#print "iter = {}".format(self.iter)
				root = 0.5*(x1+x2)
				prod = self.func(x1)*self.func(root)
				if prod < 0.: x2 = root
				else: x1 = root
				dx = x2-x1
			self.roots.append(root)
			
	def run(self):
		self.bracketRoots()
		self.bisection()
		return self.roots
		
class HCcurves:
	def __init__(self,Gamma,nT=1e13,solveTemp=True):
		self.Gamma = Gamma
		self.nT = nT
		# print("nT={}".format(self.nT))
		self.mbar = Model['mu']*Fund['mp']
		C = {}
		C['Gx_A'] = 1.5e-21
		C['Gx_B'] = 1./Model['Tx']
		C['Gc_A'] = Fund['kb']*Fund['sig_th']*Model['Tx']/(4.*np.pi*Fund['me']*Fund['c']**2)
		C['Gc_B'] = 4./Model['Tx']
		C['Lb_A'] = 3.3e-27
		C['Ll_A'] = 1.7e-18
		C['Ll_B'] = 1.3e5
		C['Ll_C'] = 1e-24
		C['Gt_A'] = Fund['sig_th']/(4.*np.pi)
		self.cgs = C
	
	# Loss and Gain functions [erg/(s gram)]
	def Lb(self,T):
		return self.units['Lb_A']*np.sqrt(T)
	
	def Ll(self,xi,T):
		return self.units['Ll_A']*np.exp(-self.units['Ll_B']/T)/(xi*np.sqrt(T)) + self.units['Ll_C']
		
	def Gt(self,xi):
		return self.units['Gt_A']*xi
		
	def Gx(self,xi,T):
		return self.units['Gx_A']*xi**0.25*(1.-self.units['Gx_B']*T)/np.sqrt(T)
	
	def Gc(self,xi,T):
		return self.units['Gc_A']*xi*(1.-self.units['Gc_B']*T)
	
	def netLosses(self,xi,T):
		return (self.Lb(T) + self.Ll(xi,T) - self.Gc(xi,T) - self.Gx(xi,T))
		
	def Gx_athena(self,xi,T):
		return self.units['Gx_A']*xi**0.25/np.sqrt(T)
	
	def solve4Equilibrium(self,xi):
		f = lambda T:self.netLosses(xi,T)
		#Numerically solve for the thermal mode using Brent's method
		return brentq(f,1e0,1e15,rtol=1e-14,maxiter=500,full_output=False,disp=True)
		
	def getEquilibriumValues(self,xi):
		self.units = self.cgs
		T0 = self.solve4Equilibrium(xi) #location on Equilibrium S-curve
		n0 = self.nT/T0
		p0 = n0*Fund['kb']*T0
		c0 = np.sqrt(self.Gamma*p0/(self.mbar*n0))
		F0 = xi*n0/(4.*np.pi)
		Omega0 = (self.Lb(T0) + self.Ll(xi,T0))*(n0/self.mbar)
		#norm = n0/self.mbar/Omega0 #4.*np.pi*F0/self.mbar/Omega0
		norm = 1./(self.Lb(T0) + self.Ll(xi,T0)) 
		
		#Physical scale of perturbation
		e0 = p0/(self.Gamma-1.)
		t_th = e0/(self.mbar*n0*Omega0)
		#Store equilibrium density
		d0 = self.mbar*n0
		return {'T0':T0,'n0':n0,'p0':p0,'c0':c0,'F0':F0,'Omega0':Omega0,'norm':norm,\
				't_th':t_th,'d0':d0}
				
	def getNonEquilbValues(self,xi,Tfac=1.,F=1.):
		self.units = self.cgs
		T0 = self.solve4Equilibrium(xi)*Tfac #location on Equilibrium S-curve
		n0 = self.nT/T0
		p0 = n0*Fund['kb']*T0
		c0 = np.sqrt(self.Gamma*p0/(self.mbar*n0))
		xi = F*xi
		F0 = xi*n0/(4.*np.pi)
		Omega0 = (self.Lb(T0) + self.Ll(xi,T0))*(n0/self.mbar)
		#norm = n0/self.mbar/Omega0 #4.*np.pi*F0/self.mbar/Omega0
		norm = 1./(self.Lb(T0) + self.Ll(xi,T0)) 
		
		#Physical scale of perturbation
		e0 = p0/(self.Gamma-1.)
		t_th = e0/(self.mbar*n0*Omega0)
		#Store equilibrium density
		d0 = self.mbar*n0
		return {'T0':T0,'n0':n0,'p0':p0,'c0':c0,'F0':F0,'Omega0':Omega0,'norm':norm,\
				't_th':t_th,'d0':d0}
		
	def setC1(self):
		self.C1 = self.getEquilibriumValues(1.)
		
	def get_rho(self,xi,Ceq,F=1.):
		self.setC1()
		T1 = self.C1['T0']
		Teq = Ceq['T0']
		F1 = self.C1['F0']
		Feq = Ceq['F0']
		return F*(Feq/F1)*(Teq/T1)/xi
	
	def get_xi(self,rho,Ceq,F=1.):
		self.setC1()
		T1 = self.C1['T0']
		Teq = Ceq['T0']
		F1 = self.C1['F0']
		Feq = Ceq['F0']
		#print "xi (200) = {}".format(F*(Feq/F1)*(Teq/T1))
		return F*(Feq/F1)*(Teq/T1)/rho
		
	#def get_Xi(self,xi,T):
	#	const = 1./(4.*np.pi*Fund['c']*Fund['kb'])
	#	return const*xi/T
		
	def get_Xi(self,rho,Ceq,T,F=1.):
		C1eq = self.setC1()
		const = 1./(4.*np.pi*Fund['c']*Fund['kb'])
		return const*self.get_xi(rho,Ceq,F)/(T*Ceq['T0'])
		
	def Gb(self,xi,T,Ceq):
		xifac = 4.*np.pi/xi/Fund['sig_th']
		Gcoeff = 1.29e25/xifac
		rho = Ceq['n0']*Fund['mp']*self.get_rho(xi,Ceq)
		return Gcoeff*rho*T**(-3.5)
		
	def get_xiFlux1(self,xis,Fac=1.):
		N = len(xis)
		xi_prime = np.zeros(N)
		for i in range(N):
			Ceq = self.getEquilibriumValues(xis[i])
			xi_prime[i] = xis[i]*Fac*Fund['c']*Ceq['n0']*Fund['kb']*Ceq['T0']/Ceq['F0']
			#Tflux[i] = self.solve4Equilibrium(xi_prime)
		return xi_prime
		
	def getSimEvolutionTracks(self,rho_bkgd,drho,Ceq,F=1.):
		C1eq = self.C1
		T1 = C1eq['T0']
		Teq = Ceq['T0']
		F1 = C1eq['F0']
		Feq = Ceq['F0']
		fac = F*(Feq/F1)*(Teq/T1)
		print("fac = {}".format(fac))
		xi = fac/rho_bkgd
		dxi = -fac*drho
		return xi,dxi
		
	def getTempTracks(self,xis,T0):
		Teq = []
		for xi in xis:
			Teq.append(self.solve4Equilibrium(xi))
		return T0/np.array(Teq)
		
	def getCodeUnits(self,Ceq,K=1.):
		e0 = Ceq['p0']/(self.Gamma-1.)
		n0 = Ceq['n0']
		T0 = Ceq['T0']
		c0 = Ceq['c0']
		Omega0 = Ceq['Omega0']
		#norm = n0/(self.mbar*Omega0)
		norm = Ceq['norm']
		sqrtT0 = np.sqrt(T0)
		#print "norm = {}, Tfac = {}".format(norm,Tfac)
		#Line-cooling coefficients
		Ll_A = self.cgs['Ll_A']*norm/sqrtT0 #/xi
		Ll_B = self.cgs['Ll_B']/T0
		Ll_C = self.cgs['Ll_C']*norm
		Consts = {'Ll_A':Ll_A, 'Ll_B':Ll_B, 'Ll_C':Ll_C} 
		#FF-cooling coefficients
		Lb_A = self.cgs['Lb_A']*norm*sqrtT0
		#Lb_B = self.Lb_coeffB/Tfac
		Consts['Lb_A'] = Lb_A
		#Compton-heating coefficients
		Gc_A = self.cgs['Gc_A']*norm #*xi
		Gc_B = self.cgs['Gc_B']*T0
		Consts['Gc_A'] = Gc_A
		Consts['Gc_B'] = Gc_B
		#Xray-heating coefficients
		Gx_A = self.cgs['Gx_A']*norm/sqrtT0 #*xi**0.25
		Gx_B = self.cgs['Gx_B']*T0
		Consts['Gx_A'] = Gx_A
		Consts['Gx_B'] = Gx_B
		#Thompson-heating coefficients
		Gt_A = self.cgs['Gt_A']*norm
		#Radiation Force
		Consts['Gt_A'] = Gt_A 
		Consts['Cratio'] = Fund['c']/c0
		#Physical scale of perturbation
		tau_th = e0/(self.mbar*n0*Omega0)
		Consts['tau_th'] = tau_th
		Consts['lambda'] = (tau_th*c0/K)
		Consts['tau_sc'] = Consts['lambda']/c0
		#Store equilibrium density
		Consts['d0'] = self.mbar*Ceq['n0']
		Consts['norm'] = norm
		#Line-driving coefficient
		v_th = 2e6
		Consts['t_A'] = n0*(c0**2.)*Fund['sig_th']*v_th/Omega0
		#Conduction 
		#Consts['kappa'] = self.getKappa(Ceq)
		Consts['CoulombLog'] = 29.7 + np.log(T0/1e6) - 0.5*np.log(n0)
		return Consts
		
	def getLengthScale(self,Ceq,K):
		Lcloud = Ceq['t_th']*Ceq['c0']/K
		return Lcloud
		
	def setCodeUnits(self,Ceq):
		self.units = self.getCodeUnits(Ceq)
		
	def resetCgsUnits(self):
		self.units = self.cgs
		
	def getL_TandL_rho(self,xi,Ceq,T=1.,rho=1.,F=1.,):
		#xi = F*xi
		Tinv = 1./T
		sqrtT = np.sqrt(T)
		
		self.setCodeUnits(Ceq)
		
		#T-derivatives
		Lb_T = 0.5*self.units['Lb_A']/sqrtT
		Ll_T = self.units['Ll_A']*np.exp(-self.units['Ll_B']*Tinv)/(xi*sqrtT)*(self.units['Ll_B']*Tinv**2. - 0.5*Tinv)
		Gx_T = -0.5*self.units['Gx_A']*xi**0.25*(Tinv+self.units['Gx_B'])/sqrtT
		Gc_T = -self.units['Gc_A']*self.units['Gc_B']*xi	
		
		#rho-derivatives: these are -xi*(dL_b/dxi + dL_l/dxi - dG_c/dx - dG_x/dxi)
		Lb_rho = 0.
		Ll_rho = self.units['Ll_A']*np.exp(-self.units['Ll_B']*Tinv)/(xi*sqrtT)
		Gx_rho = -0.25*self.units['Gx_A']*(xi**0.25)*(1.-self.units['Gx_B']*T)/sqrtT
		Gc_rho = -self.units['Gc_A']*(1.-self.units['Gc_B']*T)*xi
		#print "xi = {}, T = {}".format(xi,T)
		#print "Lb_rho = {}, Ll_rho = {}".format(Lb_rho,Ll_rho)
		#print "Gx_rho = {}, Gc_rho = {}".format(Gx_rho,Gc_rho)
		
		#net-Loss function
		netLosses = self.netLosses(xi,T)
		
		self.resetCgsUnits()
		
		#netLoss-derivatives
		L_T = rho*(Lb_T + Ll_T - Gx_T - Gc_T)
		L_rho = netLosses + (Lb_rho + Ll_rho - Gx_rho - Gc_rho)
		return L_T,L_rho
		
	def balbusDerivatives(self,xi,T):
		#xi = F*xi
		Tinv = 1./T
		sqrtT = np.sqrt(T)
		
		#T-derivatives
		Lb_T = 0.5*self.units['Lb_A']/sqrtT
		Ll_T = self.units['Ll_A']*np.exp(-self.units['Ll_B']*Tinv)/(xi*sqrtT)*(self.units['Ll_B']*Tinv**2. - 0.5*Tinv)
		Gx_T = -0.5*self.units['Gx_A']*xi**0.25*(Tinv+self.units['Gx_B'])/sqrtT
		Gc_T = -self.units['Gc_A']*self.units['Gc_B']*xi	
		
		#rho-derivatives: these are -xi*(dL_b/dxi + dL_l/dxi - dG_c/dx - dG_x/dxi)
		Lb_xi = 0.
		Ll_xi = self.units['Ll_A']*np.exp(-self.units['Ll_B']*Tinv)/(xi*sqrtT)
		Gx_xi = -0.25*self.units['Gx_A']*(xi**0.25)*(1.-self.units['Gx_B']*T)/sqrtT
		Gc_xi = -self.units['Gc_A']*(1.-self.units['Gc_B']*T)*xi
		#print "xi = {}, T = {}".format(xi,T)
		#print "Lb_rho = {}, Ll_rho = {}".format(Lb_rho,Ll_rho)
		#print "Gx_rho = {}, Gc_rho = {}".format(Gx_rho,Gc_rho)
		
		derivatives = {'Lb_T':Lb_T, 'Ll_T':Ll_T, 'Gx_T':Gx_T, 'Gc_T':Gc_T, 'Ll_xi':Ll_xi/xi, 'Gx_xi':Gx_xi/xi, 'Gc_xi':Gc_xi/xi}
		#net-Loss function
		#netLosses = self.netLosses(xi,T)
		
		#netLoss-derivatives
		#L_T = rho*(Lb_T + Ll_T - Gx_T - Gc_T)
		#L_rho = netLosses + (Lb_rho + Ll_rho - Gx_rho - Gc_rho)
		return derivatives #L_T,L_rho
		
	def getF_TandF_S(self,xi,Ceq,K=1.,T=1.,F=1.,rho=1.):
		#xi = F*xi
		Tinv = 1./T
		sqrtT = np.sqrt(T)
		self.setCodeUnits(Ceq)
		Gt_T = 0.
		Gx_T = -0.5*self.units['Gx_A']*xi**0.25*(Tinv+self.units['Gx_B'])/sqrtT
		G_T = Gt_T + Gx_T
		#rho-derivatives
		Gt_xi = self.units['Gt_A']
		Gx_xi = 0.25*self.units['Gx_A']*(xi**(-0.75))*(1.-self.units['Gx_B']*T)/sqrtT
		G_xi = Gt_xi + Gx_xi
		
		#total gains
		G = self.Gt(xi) + self.Gx(xi,T)
		self.resetCgsUnits()
		
		#terms entering the dispersion relation
		Consts = self.getCodeUnits(Ceq,K)
		const = 1./(K*Consts['Cratio']*self.Gamma*(self.Gamma-1.))
		F_T = const*(T*rho)*G_T
		F_S = const*rho*(G + T*G_T + 2.*G - xi*G_xi)
		return F_T,F_S
		
	def criterionBalbus(self,xi,Ceq,T=1.):
		#self.C1 = self.getEquilibriumValues(1.)
		L_T,L_rho = self.getL_TandL_rho(xi,Ceq,T)
		
		#const-Pressure T-derivative
		rho = self.get_rho(xi,Ceq)
		Tinv = 1./T
		#L_Tpconst = rho*L_T - rho*Tinv*L_rho
		L_Tpconst = L_T - rho*Tinv*L_rho
		
		#Evaluate net loss function
		self.setCodeUnits(Ceq)
		netLoss = rho*self.netLosses(xi,T) #note rho is here bc netLosses is only Ls - Gs
		if 1==0:
			print("rho= {}, rhs = {}".format(rho,rhs))
		self.resetCgsUnits()
		
		return (L_Tpconst - netLoss*Tinv)
		
	def badBalbus(self,xi,Ceq,T=1.):
		#self.C1 = self.getEquilibriumValues(1.)
		L_T,L_rho = self.getL_TandL_rho(xi,Ceq,T)
		
		#const-Pressure T-derivative
		rho = self.get_rho(xi,Ceq)
		Tinv = 1./T
		#L_Tpconst = rho*L_T - rho*Tinv*L_rho
		L_Tpconst = L_T - rho*Tinv*L_rho
		
		#Evaluate net loss function
		self.setCodeUnits(Ceq)
		netLoss = rho*self.netLosses(xi,T) #note rho is here bc netLosses is only Ls - Gs
		if 1==0:
			print("rho= {}, rhs = {}".format(rho,rhs))
		self.resetCgsUnits()
		
		return (L_Tpconst*T/netLoss - 1.)
		
	def solveBadBalbus(self,xi,value=0.,iroot=0,choice='isobaric'):
		Ceq = self.getEquilibriumValues(xi)
		if choice == 'isobaric':
			#f = lambda T:self.criterionBalbus(xi,Ceq,T=T) - value
			f = lambda T:self.badBalbus(xi,Ceq,T=T) - value
		elif choice == 'adiabatic':
			f = lambda T:self.criterionAdiabatic(xi,Ceq,T=T) - value
		else:
			print("Error. Unrecogized criterion choice.")
			sys.exit()
		#Employ bracketing and bisection to find any roots
		myfinder = BBrootfinder(f,minRange=0.9,maxRange=2.,Nbrackets=500)
		roots = myfinder.run()
		#root = bisect(f,1e0,2e1)
		#root = brentq(f,1e0,1e1,rtol=1e-20,maxiter=500,full_output=False,disp=True)
		#root = newton(f, 15., fprime=None, args=(), tol=1e-13, maxiter=50,fprime2=None)
		#roots = [root]
		if len(roots) == 0:
			return [np.nan]
		else:
			return roots
		
	def criterionAdiabatic(self,xi,Ceq,T=1.,DEBUG=False):
		#self.C1 = self.getEquilibriumValues(1.)
		L_T,L_rho = self.getL_TandL_rho(xi,Ceq,T)
		
		#const-Pressure T-derivative
		rho = self.get_rho(xi,Ceq)
		Tinv = 1./T
		L_Tpconst = L_T - rho*Tinv*L_rho
		
		return (L_Tpconst + 2.5*rho*Tinv*L_rho)
		
	def intersection_Balbus_Equilb(self,xi):
		Ceq = self.getEquilibriumValues(xi)
		L_T,L_rho = self.getL_TandL_rho(xi,Ceq)
		rho = self.get_rho(xi,Ceq)
		
		return L_T - rho*L_rho
		
	def intersection_Balbus_Isobaric(self,xi,xi_b):
		Ceq = self.getEquilibriumValues(xi)
		Ceq_b = self.getEquilibriumValues(xi_b)
		T = Ceq_b['T0']/Ceq['T0']
		#L_T,L_rho = self.getL_TandL_rho(xi,Ceq)
		#rho = self.get_rho(xi,Ceq)
		#netL = rho*self.netLosses(xi,T*xi/xi_b)
		#return netL + rho*L_rho - L_T*T*xi/xi_b
		self.criterionBalbus(xi,Ceq,T=T*xi/xi_b) - (T - T_b)
		#return self.criterionBalbus(xi,Ceq,Ceq_b['T0']*xi/xi_b)
		
	def sampleNetLossFunction(self,xis,Ts):
		Nxis = len(xis)
		NTs = len(Ts)
		Z = np.mat(np.zeros(Nxis*NTs)).reshape(Nxis,NTs)
		for i in range(Nxis):
			for j in range(NTs):
				Omega = self.Lb(Ts[j]) + self.Ll(xis[i],Ts[j])
				netL = self.netLosses(xis[i],Ts[j])
				Z[i,j] = netL/Omega
		return Z.T
		
	def sampleBalbusCriterion(self,xis,Ts):
		Nxis = len(xis)
		NTs = len(Ts)
		self.setC1()
		Z = np.mat(np.zeros(Nxis*NTs)).reshape(Nxis,NTs)
		for i in range(Nxis):
			Ceq = self.getEquilibriumValues(xis[i])
			Teq_inv = 1./Ceq['T0']
			for j in range(NTs):
				Z[i,j] = self.criterionBalbus(xis[i],Ceq,Ts[j]*Teq_inv)
		return Z.T
	
	def getEquilibriumDerivatives(self,xi):
		L_T,L_rho = self.getL_TandL_rho(xi)
		stability = self.criterionBalbus(xi)
		return L_T,L_rho,stability
		
	def getContourValue(self,xi,value=0.,iroot=0,choice='isobaric'):
		Ceq = self.getEquilibriumValues(xi)
		if choice == 'isobaric':
			f = lambda T:self.criterionBalbus(xi,Ceq,T=T) - value
		elif choice == 'adiabatic':
			f = lambda T:self.criterionAdiabatic(xi,Ceq,T=T) - value
		else:
			print("Error. Unrecogized criterion choice.")
			sys.exit()
		#Employ bracketing and bisection to find any roots
		myfinder = BBrootfinder(f)
		roots = myfinder.run()
		if len(roots) == 0:
			return [np.nan]
		else:
			return roots
		
	def getEquilibriumCurve(self,xis):
		Npts = len(xis)
		curve = []
		for val in xis:
			self.units = self.cgs
			curve.append(self.solve4Equilibrium(val))
		return curve
		
	def sampleTfacContour(self,xis,Ts):
		Nxis = len(xis)
		NTs = len(Ts)
		Z = np.mat(np.zeros(Nxis*NTs)).reshape(Nxis,NTs)
		for i in range(Nxis):
			Ceq = self.getEquilibriumValues(xis[i])
			Teq_inv = 1./Ceq['T0']
			for j in range(NTs):
				Z[i,j] = Ts[j]*Teq_inv
		return Z.T
		
	def contourPoints(self,xis,Ts):
		Nxis = len(xis)
		NTs = len(Ts)
		Z = np.mat(np.zeros(Nxis*NTs)).reshape(Nxis,NTs)
		for i in range(Nxis):
			if xis[i] == 1e3:
				for j in range(NTs):
					if Ts[j] == 1e5:
						Z[i,j] = 0.5
						Z[i+1,j] = 1.
						Z[i-1,j] = 1.5
		return Z.T
		
	def getEquilibriumComponents(self,xis,Ts):
		Nvals = len(Ts)
		if len(xis) != Nvals:
			print("Error: arrays must correspond to each other!")
			sys.exit()
		Gx = []; Gc = []; Lb = []; Ll = []
		for i in range(Nvals):
			n = self.nT/Ts[i]
			n2 = n**2
			Gx.append(n2*self.Gx(xis[i],Ts[i]))
			Gc.append(n2*self.Gc(xis[i],Ts[i]))
			Lb.append(n2*self.Lb(Ts[i]))
			Ll.append(n2*self.Ll(xis[i],Ts[i]))
		components = {'Gx':Gx, 'Gc':Gc, 'Lb':Lb, 'Ll':Ll}
		return components
		
	def getDerivativeContours(self,xis,value=0.,choice='isobaric'):
		Nvals = len(xis)
		contour1 = []
		contour2 = []
		self.C1 = self.getEquilibriumValues(1.)
		for i in range(Nvals):
			print("xi = {}".format(xis[i]))
			points = self.getContourValue(xis[i],value,choice)
			contour1.append(points[0])
			if len(points)>1:
				contour2.append(points[1])
			else:
				contour2.append(float('NaN'))
		return contour1,contour2
	
	def getDerivativeCurves(self,xis,Tfac=1.,TIMESCALES = True):
		Nvals = len(xis)
		self.C1 = self.getEquilibriumValues(1.)
		L_Ts = []; L_rhos = []; values = []
		for i in range(Nvals):
			Ceq = self.getEquilibriumValues(xis[i])
			L_T,L_rho = self.getL_TandL_rho(xis[i],Ceq)
			stability = self.criterionBalbus(xis[i],Ceq,DEBUG=False)
			L_Ts.append(L_T)
			L_rhos.append(L_rho)
			if TIMESCALES:
				values.append(1./stability) #timescale is 1./(stability criterion)
			else: values.append(stability)
		return L_Ts,L_rhos,values
		
	def getNegRanges(self,xis,Tfac):
	# For each Tfac, extract the unstable range of xis from the xis array and
	# record the min/max of this range
		Nxis = len(xis)
		negRange = []
		L_T,L_rho,stability = self.getDerivativeCurves(xis,Tfac)
		negIndex = [i if stability[i] < 0. else -2 for i in range(Nxis)]
		changes = [negIndex[i+1] - negIndex[i] for i in range(Nxis-1)]
		begIndex = []
		endIndex = []
		for i in range(Nxis-1):
			if changes[i] > 1:
				begIndex.append(i+1)
			if changes[i] < -2:
				endIndex.append(i)
		#print "nedIndex = {}".format(negIndex)
		#print "changes = {}".format(changes)
		#print "begIndex = {}, endIndex = {}".format(begIndex,endIndex)
		ranges = []
		Nranges = max(len(begIndex),len(endIndex))
		#Append the last value of xi if ranges don't match
		if len(endIndex) < Nranges:
				endIndex.append(Nxis-1)
		if len(begIndex) < Nranges:
			begIndex.append(0)
		for j in range(Nranges):
			xirange = xis[begIndex[j]:endIndex[j]]
			ranges.append(xirange)
		return ranges
		
	def getMinMaxXis(self,xis,Tfacs):
	# For each Tfac, extract the unstable range of xis from the xis array and
	# record the min/max of this range
		NTfacs = len(Tfacs)
		mins1 = np.zeros(NTfacs)
		maxs1 = np.zeros(NTfacs)
		mins2 = np.zeros(NTfacs)
		maxs2 = np.zeros(NTfacs)
		Nxis = len(xis)
		for j in range(NTfacs):
			xi_ranges = self.getNegRanges(xis,Tfacs[j])
			xi_range = xi_ranges[0]
			if len(xi_range) > 0: #TRY TO AVOID THIS!
				mins1[j] = min(xi_range)
				maxs1[j] = max(xi_range)
			
			Nregions = len(xi_ranges)
			print("Tfac = {}, Nregions = {}".format(Tfacs[j],Nregions))
			if Nregions > 1:
				xi_range = xi_ranges[1]
				mins2[j] = min(xi_range)
				maxs2[j] = max(xi_range)
		return mins1,maxs1,mins2,maxs2		
		
	def findEndStates(self,xi_b,T_b):
		#(xi_b,T_b) is a starting point on the equilibrium curve
		#return xi locations connected to xi_b by an isobaric line
		f = lambda xi:self.netLosses(xi,T_b*xi/xi_b)
		#Employ bracketing and bisection to find any roots
		myfinder = BBrootfinder(f,minRange=1.,maxRange=1e4,Nbrackets=2000,tolerance=1e-12)
		roots = myfinder.run()
		if len(roots) == 0:
			return [np.nan]
		else:
			return roots
	
	def findEdgesOfGreyRegion(self):
		#T'*L_T - rho'*L_rho = 0 define the boundaries of unstable/stable regions on S-curve
		f = lambda xi:self.intersection_Balbus_Equilb(xi)
		#Employ bracketing and bisection to find any roots
		myfinder = BBrootfinder(f,minRange=1.,maxRange=1e4,Nbrackets=2000,tolerance=1e-12)
		roots = myfinder.run()
		if len(roots) == 0:
			return [np.nan]
		else:
			return roots
			
	def findStablePoints(self,xi_b):
		#T'*L_T - rho'*L_rho = 0 define the boundaries of unstable/stable regions on S-curve
		#f = lambda xi:self.intersection_Balbus_Isobaric(xi,xi_b) #self.criterionBalbus(xi,Ceq,xi/xi_b)
		f = lambda xi:self.criterionBalbus(xi,T=xi/xi_b)
		#Employ bracketing and bisection to find any roots
		myfinder = BBrootfinder(f,minRange=1e2,maxRange=1e4,Nbrackets=2000,tolerance=1e-12)
		roots = myfinder.run()
		if len(roots) == 0:
			return [np.nan]
		else:
			return roots
			
	def getKappa(self,eqvals,pow):
		T = eqvals['T0'] 
		n = eqvals['n0']
		CoulombLog = 29.7 + np.log((T/1e6)/np.sqrt(n))
		psi = 1.84e-5/CoulombLog
		return psi*T**pow
	
	def getFieldLength(self,eqvals,p=2.5):
		kappa = self.getKappa(eqvals,p)
		T = eqvals['T0']
		rho = eqvals['d0']
		Omega = eqvals['Omega0']
		return 2.*np.pi*np.sqrt(kappa*T/(Omega*rho))
		
class TImodes:
	def __init__(self,params,Conduction=False,Equilibrium=True,DEBUG=False):
		#self.tau_th = params['tau_th']
		self.g = params['gamma']
		self.k = params['k']
		self.K = params['K']
		self.M0 = params['M0']
		#self.cs = params['cs']
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
# 	def setCubicField(self):
# 		if self.DEBUG:
# 			print "======================"
# 			print "    Field's cubic     "
# 			print "======================"
# 		self.a = self.L_T/self.tau_th
# 		self.b = (self.k*self.cs)**2.
# 		self.c = (self.L_T-self.L_rho)*self.b/(self.g*self.tau_th) 
# 		self.prepareCubic()
# 		kcs = self.k*self.cs
# 		denom = (kcs/self.g)**2.
# 		Zreal = 0.
# 		Zimag = 0.
# 		self.Z = complex(Zreal,Zimag)
# 		Wreal = 1./(self.g*denom)
# 		Wimag = 0.
# 		self.W = complex(Wreal,Wimag)
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
		kcs = self.k*self.cs
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
	def pressureAmplitudes(self,p0,nR,nI):
		#Areal = p0*(self.Z.real + (nI**2 - nR**2)*self.W.real + 2.*nR*nI*self.W.imag)
		#Areal = -nR**2*self.W.real
		#Aimag = p0*(self.Z.imag + (nI**2 - nR**2)*self.W.imag - 2.*nR*nI*self.W.real)
		Preal = self.g*((nI**2 - nR**2)/self.k**2 + 2.*nI*self.M0/self.k + self.M0**2)
		Pimag = -2.*self.g*nR*(nI/self.k + self.M0)/self.k
		return Preal,Pimag
	def cubicFunc(self,n):
		x = n+self.n0
		return x**3 + self.a*x**2 + self.b*x + self.c
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
	def thermalTime(self,g,n):
		tau = (1. - 0.5*(1. + g*n**2/self.b))/(g*n*(1. + n**2/self.b))
		return tau
		
class customSubplots:
	def __init__(self,Nplots,DEBUG=False):
		self.Nplots = Nplots
	def init_lines(self,Nlines=12):
		xdata = []; ydata = []; clr = []; mrk = []; sty = []; lab = []; lwd = []
		for l in range(Nlines):
			xdata.append([])
			ydata.append([])
			clr.append(['k'])
			mrk.append([''])
			sty.append(['-'])
			lab.append(['_nolegend_'])
			lwd.append([1])
		return xdata, ydata, clr, mrk, sty, lwd, lab
	def assignSubplots(self,fig,xlabel='x'):
		subplots = []
		if self.Nplots == 1:
			subplots.append(fig.add_subplot(111))
			subplots[0].set_xlabel(xlabel)
		elif self.Nplots == 2:
			#subplots.append(fig.add_subplot(211))
			#subplots.append(fig.add_subplot(212))
			gs = gridspec.GridSpec(1,1) #makes 1 row, 1 column region for placing subplots
			gs0 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[0])
			ax1 = plt.Subplot(fig, gs0[0:-2,:])
			ax2 = plt.Subplot(fig, gs0[1:,:])
			subplots.append(fig.add_subplot(ax1))
			subplots.append(fig.add_subplot(ax2))
			#ax2.set_xlabel(xlabel)
		elif self.Nplots == 3:
			subplots.append(fig.add_subplot(311))
			subplots.append(fig.add_subplot(312))
			subplots.append(fig.add_subplot(313))
			subplots[2].set_xlabel(xlabel)
		elif self.Nplots == 4:
			subplots.append(fig.add_subplot(411))
			subplots.append(fig.add_subplot(412))
			subplots.append(fig.add_subplot(413))
			subplots.append(fig.add_subplot(414))
			plt.setp(subplots[0].get_xticklabels(), visible=False)
			plt.setp(subplots[1].get_xticklabels(), visible=False)
			plt.setp(subplots[2].get_xticklabels(), visible=False)
		if self.Nplots == 6:
			subplots.append(fig.add_subplot(611))
			subplots.append(fig.add_subplot(612))
			subplots.append(fig.add_subplot(613))
			subplots.append(fig.add_subplot(614))
			subplots.append(fig.add_subplot(615))
			subplots.append(fig.add_subplot(616))
			#subplots[2].set_xlabel(xlabel)
			#subplots[5].set_xlabel(xlabel)
		if self.Nplots == 7:
			subplots.append(fig.add_subplot(711))
			subplots.append(fig.add_subplot(712))
			subplots.append(fig.add_subplot(713))
			subplots.append(fig.add_subplot(714))
			subplots.append(fig.add_subplot(715))
			subplots.append(fig.add_subplot(716))
			subplots.append(fig.add_subplot(717))
			#subplots[2].set_xlabel(xlabel)
			#subplots[6].set_xlabel(xlabel)
		elif self.Nplots == 8:
			subplots.append(fig.add_subplot(811))
			subplots.append(fig.add_subplot(812))
			subplots.append(fig.add_subplot(813))
			subplots.append(fig.add_subplot(814))
			subplots.append(fig.add_subplot(815))
			subplots.append(fig.add_subplot(816))
			subplots.append(fig.add_subplot(817))
			subplots.append(fig.add_subplot(818))
			#subplots[3].set_xlabel(xlabel)
			#subplots[7].set_xlabel(xlabel)
		elif 1==0 and self.Nplots == 10:
			gs0 = gridspec.GridSpec(1, 3) #makes 1 row, 3 column region for placing subplots
			gsL = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0]) #allocate space for 3x3 plot region in first column of gs0
			gsM = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[1])
			gsR = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[2])
			#define left column layout
			ax1 = plt.Subplot(fig, gsL[0,:]) 
			ax2 = plt.Subplot(fig, gsL[1,:])
			ax3 = plt.Subplot(fig, gsL[2,:]) 
			ax4 = plt.Subplot(fig, gsL[3,:])
			#define right column layout
			ax5 = plt.Subplot(fig, gsR[0,:]) 
			ax6 = plt.Subplot(fig, gsR[1,:]) 
			ax7 = plt.Subplot(fig, gsR[3:-1,:]) 
			#define middle column layout
			ax8 = plt.Subplot(fig, gsM[0:-2,:])
			ax9 = plt.Subplot(fig, gsM[2,:])
			ax10 = plt.Subplot(fig, gsM[3,:])
			#gs0.tight_layout(fig, h_pad=0.)
			#assign subplots to figure
			subplots.append(fig.add_subplot(ax1))
			subplots.append(fig.add_subplot(ax2))
			subplots.append(fig.add_subplot(ax3))
			subplots.append(fig.add_subplot(ax4))
			subplots.append(fig.add_subplot(ax5))
			subplots.append(fig.add_subplot(ax6))
			subplots.append(fig.add_subplot(ax7))
			subplots.append(fig.add_subplot(ax8))
			subplots.append(fig.add_axes([0.405,0.74, 0.1, 0.15])) #inset TL
			subplots.append(fig.add_axes([0.52,0.525, 0.1, 0.15])) #inset BR
			subplots.append(fig.add_subplot(ax9))
			subplots.append(fig.add_subplot(ax10))
			subplots[3].set_xlabel(xlabel)
			subplots[6].set_xlabel(xlabel)
			#remove tick-labels from insets
			plt.setp(subplots[8].get_xticklabels(), visible=False)
			plt.setp(subplots[8].get_yticklabels(), visible=False)
			plt.setp(subplots[9].get_xticklabels(), visible=False)
			plt.setp(subplots[9].get_yticklabels(), visible=False)
		elif self.Nplots == 10:
			gs0 = gridspec.GridSpec(1, 3) #makes 1 row, 3 column region for placing subplots
			gsL = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0]) #allocate space for 3x3 plot region in first column of gs0
			gsM = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[1])
			gsR = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[2])
			#define left column layout
			ax1 = plt.Subplot(fig, gsL[0,:]) 
			ax2 = plt.Subplot(fig, gsL[1,:])
			ax3 = plt.Subplot(fig, gsL[2,:]) 
			ax4 = plt.Subplot(fig, gsL[3,:])
			#define right column layout
			ax5 = plt.Subplot(fig, gsR[0,:]) 
			ax6 = plt.Subplot(fig, gsR[1,:]) 
			ax7 = plt.Subplot(fig, gsR[2,:]) 
			ax8 = plt.Subplot(fig, gsR[3,:]) 
			#define middle column layout
			ax9 = plt.Subplot(fig, gsM[0:-2,:])
			ax10 = plt.Subplot(fig, gsM[3:-1,:]) 
			#gs0.tight_layout(fig, h_pad=0.)
			#assign subplots to figure
			subplots.append(fig.add_subplot(ax1))
			subplots.append(fig.add_subplot(ax2))
			subplots.append(fig.add_subplot(ax3))
			subplots.append(fig.add_subplot(ax4))
			subplots.append(fig.add_subplot(ax5))
			subplots.append(fig.add_subplot(ax6))
			subplots.append(fig.add_subplot(ax7))
			subplots.append(fig.add_subplot(ax8))
			subplots.append(fig.add_subplot(ax9))
			subplots.append(fig.add_axes([0.405,0.74, 0.1, 0.15])) #inset TL
			subplots.append(fig.add_axes([0.52,0.525, 0.1, 0.15])) #inset BR
			subplots.append(fig.add_subplot(ax10))
			subplots[3].set_xlabel(xlabel)
			subplots[7].set_xlabel(xlabel)
			subplots[11].set_xlabel(xlabel)
			#remove tick-labels from insets
			plt.setp(subplots[9].get_xticklabels(), visible=False)
			plt.setp(subplots[9].get_yticklabels(), visible=False)
			plt.setp(subplots[10].get_xticklabels(), visible=False)
			plt.setp(subplots[10].get_yticklabels(), visible=False)
		elif self.Nplots == 11:
			gs0 = gridspec.GridSpec(1, 3) #makes 1 row, 3 column region for placing subplots
			gsL = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[0]) #allocate space for 3x3 plot region in first column of gs0
			gsM = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[1])
			gsR = gridspec.GridSpecFromSubplotSpec(4, 3, subplot_spec=gs0[2])
			#define left column layout
			ax1 = plt.Subplot(fig, gsL[0,:]) 
			ax2 = plt.Subplot(fig, gsL[1,:])
			ax3 = plt.Subplot(fig, gsL[2,:]) 
			ax4 = plt.Subplot(fig, gsL[3,:])
			#define right column layout
			ax5 = plt.Subplot(fig, gsR[0,:]) 
			ax6 = plt.Subplot(fig, gsR[1,:]) 
			ax7 = plt.Subplot(fig, gsR[2,:]) 
			ax8 = plt.Subplot(fig, gsR[3,:]) 
			#define middle column layout
			ax9 = plt.Subplot(fig, gsM[0:-2,:])
			ax10 = plt.Subplot(fig, gsM[2,:])
			ax11 = plt.Subplot(fig, gsM[3,:])
			#gs0.tight_layout(fig, h_pad=0.)
			#assign subplots to figure
			subplots.append(fig.add_subplot(ax1))
			subplots.append(fig.add_subplot(ax2))
			subplots.append(fig.add_subplot(ax3))
			subplots.append(fig.add_subplot(ax4))
			subplots.append(fig.add_subplot(ax5))
			subplots.append(fig.add_subplot(ax6))
			subplots.append(fig.add_subplot(ax7))
			subplots.append(fig.add_subplot(ax8))
			subplots.append(fig.add_subplot(ax9))
			subplots.append(fig.add_axes([0.405,0.74, 0.1, 0.15])) #inset TL
			subplots.append(fig.add_axes([0.52,0.525, 0.1, 0.15])) #inset BR
			subplots.append(fig.add_subplot(ax10))
			subplots.append(fig.add_subplot(ax11))
			subplots[3].set_xlabel(xlabel)
			subplots[7].set_xlabel(xlabel)
			#remove tick-labels from insets
			plt.setp(subplots[9].get_xticklabels(), visible=False)
			plt.setp(subplots[9].get_yticklabels(), visible=False)
			plt.setp(subplots[10].get_xticklabels(), visible=False)
			plt.setp(subplots[10].get_yticklabels(), visible=False)
		return subplots
	def setPlotLimitsX(self,data,plot,xmin=0.,xmax=0.):
		if xmin == 0.:
			imin = data.argmin()
			min = data[imin]
			xmin = min
		if xmax == 0.:
			imax = data.argmax()
			max = data[imax]
			xmax = max
		#print "min,max = {},{}".format(xmin,xmax)
		plot.set_xlim(xmin,xmax)
	def setAllLimitsX(self,data,subplots):
		for i in range(self.Nplots):
			self.setPlotLimitsX(data,subplots[i])
	def setPlotLimitsY(self,data,plot,zoom=0.,data2=None,Nticks=5,CUSTOM_TICKS=False):
		imax = data.argmax()
		imin = data.argmin()
		avg = np.mean(data)
		min = data[imin]
		max = data[imax]
		Amin = abs(min - avg)
		Amax = abs(max - avg)
		Aavg = 0.5*(Amin + Amax)
		amp = Aavg
		ymin = min
		ymax = max
		if data2 != None:
			imax = data2.argmax()
			imin = data2.argmin()
			avg = np.mean(data2)
			min = data2[imin]
			max = data2[imax]
			Amin = abs(min - avg)
			Amax = abs(max - avg)
			Aavg = 0.5*(Amin + Amax)
			if min < ymin:
				ymin = min
			if max > ymax:
				ymax = max
			amp = np.max(amp,Aavg)
			#print "2: min,max = {},{}".format(ymin,ymax)
		ymin = ymin-zoom*amp
		ymax = ymax+zoom*amp
		plot.set_ylim(ymin,ymax)
		if CUSTOM_TICKS: #Custom tick formatting
			dy = (ymax-ymin)/Nticks
			majorLocator   = MultipleLocator(dy)
			majorFormatter = FormatStrFormatter('%.2e')
			minorLocator   = MultipleLocator(.1*dy)
			plot.yaxis.set_major_locator(majorLocator)
			plot.yaxis.set_major_formatter(majorFormatter)
			#for the minor ticks, use no labels; default NullFormatter
			plot.yaxis.set_minor_locator(minorLocator)
	def setAdaptiveLimitsY(self,allVariableData,subplots,pad):
		Nvars = len(allVariableData)
		#if Nvars < self.Nplots:
	#		print "Fewer variables then plots!"
	#		sys.exit()
		for i in range(Nvars):
			dataArray = allVariableData[i]
			Ndata = len(dataArray)
			if Ndata > 2:
				print("ERROR: ONLY 2 SETS OF DATA ALLOWED CURRENTLY!")
				sys.exit()
			elif Ndata == 2:
				data = dataArray[0]
				data2 = dataArray[1]
			else:
				data = dataArray[0]
				data2 = None
			self.setPlotLimitsY(data,subplots[i],pad,data2)
	
	def customTicks(self,p,subplots,ymin,ymax,amp,Nticks):			
			#Custom tick formatting
			dy = (ymax-ymin)/Nticks
			dy1 = 10.**np.ceil(np.log10(dy))
			dy = 10.**np.floor(np.log10(dy))
			
			#if (max(ymax/dy1,ymax/dy2)) > 10.:
			#	dy = dy1
			#else:
			#	dy = dy2
			if 1==0:
				print("A: var = {}, min = {}, max = {}, dy = {}, ymax/dy = {}, amp/dy = {} Nticks = {}".format(var,ymin,ymax,dy,ymax/dy,amp/dy,(ymax-ymin)/dy))
				
			while(2*amp/dy) > 10.:
				dy *= 10.
			
			if amp/dy > 4.5:
				ymax += amp
				ymin -= amp
				dy *= 10.
				
			#while(ymax/dy) > 10.:
			#	dy *= 10.
			
			#round ymin and ymax to nearest power of 10
# 			if ymin < 0.:
# 				ymin =  -10.**np.floor(np.log10(abs(ymin)))
# 			else:
# 				ymin =  10.**np.floor(np.log10(ymin))
# 			if ymax < 0.:
# 				ymax =  -10.**np.ceil(np.log10(abs(ymax)))
# 			else:
# 				ymax =  10.**np.ceil(np.log10(abs(ymax)))
				
			Nticks = (ymax-ymin)/dy
			while(Nticks > 12.):
				dy *= 10.
				Nticks = (ymax-ymin)/dy
				
			if 1==0:
				print("B: var = {}, min = {}, max = {}, dy = {}, ymax/dy = {}, amp/dy = {}, Nticks = {}".format(var,ymin,ymax,dy,ymax/dy,amp/dy,(ymax-ymin)/dy))
				
			majorLocator   = MultipleLocator(dy)
			majorFormatter = FormatStrFormatter('%.1e')
			minorLocator   = MultipleLocator(.1*dy)
			subplots[p].yaxis.set_major_locator(majorLocator)
			subplots[p].yaxis.set_major_formatter(majorFormatter)
			#for the minor ticks, use no labels; default NullFormatter
			subplots[p].yaxis.set_minor_locator(minorLocator)
			
	def setGlobalPlotLimitsY(self,subplots,vars,ymins0,ymaxs0,yminsN,ymaxsN,zoom=0.,Nticks=5,CUSTOM_TICKS=False):
		for p in range(self.Nplots):
			var = vars[p]
			ymin = min(ymins0[var],yminsN[var])
			ymax = max(ymaxs0[var],ymaxsN[var])
			amp = ymaxs0[var] - ymins0[var]
			
			ymin = ymin-zoom*amp
			ymax = ymax+zoom*amp
			
			if CUSTOM_TICKS:
				self.customTicks(p,subplots,ymin,ymax,amp,Nticks)
			
			subplots[p].set_ylim(ymin,ymax)
	
	def setGlobalPlotLimits(self,subplots,vars,ymins0,ymaxs0,yminsN,ymaxsN,ymins0b,ymaxs0b,yminsNb,ymaxsNb,zoom=0.,Nticks=5,CUSTOM_TICKS=True):
		for p in range(len(vars)):
			var = vars[p]
			ymin = min(ymins0[var],yminsN[var])
			ymax = max(ymaxs0[var],ymaxsN[var])
			amp = ymaxs0[var] - ymins0[var]
			
			ymin2 = min(ymins0b[var],yminsNb[var])
			ymax2 = max(ymaxs0b[var],ymaxsNb[var])
			amp2 = ymaxs0b[var] - ymins0b[var]
			
			ymin = min(ymin,ymin2)
			ymax = max(ymax,ymax2)
			amp = max(amp,amp2)
			
			ymin = ymin-zoom*amp
			ymax = ymax+zoom*amp
			
			if var!='velocity' and var!='Mach number' and var!='agas':
				ymin = max(0.,ymin)
			
			if CUSTOM_TICKS:
				self.customTicks(p,subplots,ymin,ymax,amp,Nticks)
			
			subplots[p].set_ylim(ymin,ymax)
			
	def setYLabels(self,keys,subplots):
		for i in range(self.Nplots):
			#subplots[i].set_ylabel(keys[i])
			subplots[i].set_title(keys[i])
	def removeXTicks(self,subplots):
		if self.Nplots == 1:
			indices = []
		if self.Nplots == 2:
			indices = [0]
		elif self.Nplots == 3:
			indices = [0,1]
		elif self.Nplots == 4:
			indices = [0,1,2,3]
			for i in indices:
				plt.setp(subplots[i].get_yticklabels(), visible=False)
		elif self.Nplots == 6:
			indices = [0,1,2,3,4,5]
			for i in indices:
				plt.setp(subplots[i].get_yticklabels(), visible=False)
		elif self.Nplots == 7:
			indices = [0,1,2,3,4,5]
		elif self.Nplots == 8:
			indices = [0,1,2,3,4,5,6]
		elif self.Nplots == 10:
			indices = [0,1,2,4,5,6,9]
		elif self.Nplots == 11:
			indices = [0,1,2,4,5,6,9,11]
		for i in indices:
			plt.setp(subplots[i].get_xticklabels(), visible=False)
			
	def assignData(self,i,subplots,items,xdata,ydata,colors,markers,linestyles,linewidths=None,logx=False,logy=False,labels=None):
		Nlines = len(subplots[i].lines)
		#clear existing lines
		for n in range(Nlines):
			#print Nlines,i,n
			del subplots[i].lines[0] #stay at the bottom of the stack! 
		#add new lines to subplot[i]
		for l in items:
			if labels == None:
				mylabel = '_nolegend_'
			else:
				mylabel = labels[l]
			if not(logx) and not(logy):
				line, = subplots[i].plot([],[],label=mylabel)
			elif logx and not(logy):
				line, = subplots[i].semilogx([],[],label=mylabel)
			elif not(logx) and logy:
				line, = subplots[i].semilogy([],[],label=mylabel)
			else:
				line, = subplots[i].loglog([],[],label=mylabel)
			#print "lines={}, markers={}".format(lines[l],markers[l])
			line.set_data(xdata[l],ydata[l])
			line.set_color(colors[l])
			line.set_marker(markers[l])
			if linestyles[l] == '---':
				line.set_linestyle('--')
				line.set_dashes([4, 2]) #short dashed line: 8 pixels of dashed line, 4 pixels blank
			else:
				line.set_linestyle(linestyles[l])
			if linewidths != None:
				line.set_linewidth(linewidths[l])
		
	def appendData(self,i,subplots,items,xdata,ydata,colors,markers,linestyles,linewidths=None,logx=False,logy=False,labels=None):
		Nlines = len(subplots[i].lines)
		#add new lines to subplot[i]
		for l in items:
			if labels == None:
				mylabel = '_nolegend_'
			else:
				mylabel = labels[l]
			if not(logx) and not(logy):
				line, = subplots[i].plot([],[],label=mylabel)
			elif logx and not(logy):
				line, = subplots[i].semilogx([],[],label=mylabel)
			elif not(logx) and logy:
				line, = subplots[i].semilogy([],[],label=mylabel)
			else:
				line, = subplots[i].loglog([],[],label=mylabel)
			#print "lines={}, markers={}".format(lines[l],markers[l])
			line.set_data(xdata[l],ydata[l])
			line.set_color(colors[l])
			line.set_marker(markers[l])
			if linestyles[l] == '---':
				line.set_linestyle('--')
				line.set_dashes([4, 2]) #short dashed line: 8 pixels of dashed line, 4 pixels blank
			else:
				line.set_linestyle(linestyles[l])
			if linewidths != None:
				line.set_linewidth(linewidths[l])
			
	def addZoomBoxes(self,i,subplots,xi_box,T_box,pad,Npts,hotcold_shadings,zfac=1.):
		#first clear previous boxes from the main plot
		for coll in (subplots[i].collections):
			delete = True
			#don't clear the hot-cold fill_region
			for shading in hotcold_shadings:
				if shading == coll:
					delete = False
			if delete:
				subplots[i].collections.remove(coll)
		#this must be done twice for some reason
		for coll in (subplots[i].collections):
			delete = True
			#don't clear the hot-cold fill_region
			for shading in hotcold_shadings:
				if shading == coll:
					delete = False
			if delete:
				subplots[i].collections.remove(coll)
		
		ones = np.ones(Npts)
		box1limits = {}
		box2limits = {}
		limits = [box1limits,box2limits]
		#add boxes to the main plot (one for hot, one for cold)
		for ibox in range(len(xi_box)):
			if ibox == 0:
				pad_lo, pad_hi = 0.2*pad,pad
			else:
				pad_lo, pad_hi = pad,0.2*pad
			logxmin = np.log10(xi_box[ibox]) - pad_lo
			logxmax = np.log10(xi_box[ibox]) + pad_hi
			logymin = np.log10(T_box[ibox]) - pad_lo
			logymax = np.log10(T_box[ibox]) + pad_hi
			
			xis = np.logspace(logxmin,logxmax,Npts)
			bottom_region = 10**logymin*ones
			top_region = 10**logymax*ones
			subplots[i].fill_between(xis,bottom_region,top_region,facecolor='black',alpha=0.25)
			
			# size of grey boxes based on pad
			pad_lo, pad_hi = pad_lo*zfac, pad_hi*zfac #minus 1 since already zoomed in by 1 factor of pad
			logxmin = np.log10(xi_box[ibox]) - pad_lo
			logxmax = np.log10(xi_box[ibox]) + pad_hi
			logymin = np.log10(T_box[ibox]) - pad_lo
			logymax = np.log10(T_box[ibox]) + pad_hi
			limits[ibox]['xmin'] = 10**logxmin
			limits[ibox]['xmax'] = 10**logxmax
			limits[ibox]['ymin'] = 10**logymin
			limits[ibox]['ymax'] = 10**logymax
			#print "limits = {}".format(limits)
			
		return limits
		
			
#---------------------------------------------------------------------
#                   $$$ Constants used for PW14 $$$
#---------------------------------------------------------------------
Fund = {'mp':1.672622e-24, 'me':9.1093898e-28, 'c':3.0e10, 'h':6.6260755e-27, \
				'e':4.8032068e-10, 'kb':1.380658e-16, 'G':6.67259e-8}
ThomsonCrossSection = (8.*np.pi/3.)*(Fund['e']**2/(Fund['me']*Fund['c']**2))**2
Fund['sig_th'] = ThomsonCrossSection
Model = {'mu':1., 'Tx':1.16e8, 'M_BH':1.99e41}
EddingtonLuminosity = 4.*np.pi*Fund['G']*Model['M_BH']*Fund['mp']*Fund['c']/Fund['sig_th']
Model['L_edd'] = EddingtonLuminosity
R_ISCO = 6.*Fund['G']*Model['M_BH']/Fund['c']**2
Model['R_ISCO'] = R_ISCO

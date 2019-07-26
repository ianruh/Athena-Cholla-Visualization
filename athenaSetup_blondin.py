from PW15classes import *

""" Inputs """
TC = 0 # flag to include thermal conduction
xi_b = 190.
nT = 1e13
Gamma = 5./3.

#perturbation properties
xo = 0.; xf = 1.
lx = (xf - xo)
kx = 2.*np.pi/lx

# Properties of unperturbed state 
P0 = 1./Gamma
d0 = 1.
cs2 = Gamma*P0/d0
cs0 = np.sqrt(cs2)
M0 = 0.	# initial Mach number
t_th = 1.			# ratio of thermal to sound crossing time


# H/C derivatives
hcModel = blondin.HCcurves(Gamma,nT=nT)
eqvals = hcModel.getEquilibriumValues(xi_b)
athena = hcModel.getCodeUnits(eqvals,1.)
L_T,L_rho = hcModel.getL_TandL_rho(xi_b,eqvals)
print "L_T = {}, L_rho = {}".format(L_T,L_rho)

# Conduction
field_length = getFieldLength(eqvals)/athena['lambda']
gm1 = Gamma - 1.
kappa_iso = (0.5*field_length/np.pi)**2*(1./t_th)*(1./(Gamma*gm1));

""" Calculate coefficients for ICs """

TIparams_x = {'gamma':Gamma, 'cs0':cs0, 'k':kx, 'K':t_th, 'L_T':L_T, 'L_rho':L_rho, \
			'lambdaF':field_length ,'netLosses':0., 'M0':M0}

TIparams_x['F_T'] = 0.
TIparams_x['F_S'] = 0.
RFparams = {'alpha':0., 'beta':0., 'tau_rad':1.}

coeffs_x = getCoeffs(TIparams_x,RFparams,P0,TC=TC)


#===============================================
print "\nPASTE INTO ATHENA PROBLEM FILE:\n"
print "/***********************************/"
print "#define nReal " + `coeffs_x['nR']`
print "#define nImag " + `coeffs_x['nI']`
print "#define pReal " + `coeffs_x['Ap_r']`
print "#define pImag " + `coeffs_x['Ap_i']`
print "#define M0 " + `M0`
print "/***********************************/\n\n"

print "\n----------------------------------"
print "lambda_x = {}".format(lx)
print "kappa_iso = {}".format(kappa_iso)
print "T0 = {}".format(eqvals['T0'])
print "xi0 = {}".format(xi_b)
print "nR = {}".format(coeffs_x['nR'])
print "nI = {}".format(coeffs_x['nI'])
print "pR = {}".format(coeffs_x['Ap_r']*Gamma)
print "pI = {}".format(coeffs_x['Ap_i'])
print "\n----------------------------------"
print "Amplitude Requirement:"
Abound = (Gamma/cs2)*coeffs_x['Ap_r']
print "Abound = {}, gamma(n/k)^2/cs^2 = {}".format(Abound,Gamma*(coeffs_x['nR']/kx)**2/cs2)
print "Density perturbation should be << {} for linear theory to hold!!".format(abs(Abound)) 
print "----------------------------------\n"

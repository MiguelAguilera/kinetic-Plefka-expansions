import numpy as np
from scipy.integrate import quad, cumtrapz
from itertools import combinations

def bool2int(x):  # Transform bool array into positive integer
    y = 0
    for i, j in enumerate(np.array(x)[::-1]):
        y += j * 2**i
    return y


def bitfield(n, size):  # Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0] * (size - len(x)) + x
    return np.array(x)


class ising:
	def __init__(self, netsize):  # Create ising model

		self.size = netsize
		self.H = np.zeros(netsize)
		self.J = np.zeros((netsize, netsize))
		self.Beta = 1
		self.set_integration_method('FastMF')
		self.randomize_state()
		self.T=100

	def set_integration_method(self, method,T=100):
		if method not in ['MC', 'MF', 'FastMF']:
			print('Method should be one of the following: MC, MF, FastMF')
			exit(-1)
		self.integration_method = method
		self.T=T

	def randomize_state(self):
		self.s = np.random.randint(0, 2, self.size) * 2 - 1
		self.h = self.H + np.dot(self.J,self.s)

	def random_fields(self):  # Set random values for h
		self.H = np.random.rand(self.size) * 2 - 1
	def random_wiring(self):  # Set random values for J
		self.J = np.random.randn(self.size, self.size)/self.size  # /float(self.size)

	def GlauberStep(self):

		self.h = self.H + np.dot(self.J,self.s)
		r = np.random.rand(self.size)
		self.s = -1 + 2 * (2 * self.Beta * self.h > -np.log(1 / r - 1)).astype(int)

	def set_mechanism_purview(self,pc,pf):
		self.mechanism = list(pc)
		self.purview = list(pf)
		
		M = np.zeros((self.size, self.size))
		M[np.ix_(pf, pc)] = 1
		self.Jext = self.J * (1 - M)
		self.Jint = self.J * M
	
	def set_partition(self, pc1, pc2, pf1, pf2):

		pc = np.concatenate((pc1, pc2)).astype(int)
		pf = np.concatenate((pf1, pf2)).astype(int)
		self.set_mechanism_purview(pc,pf)
		
		self.partition = (pc1, pc2, pf1, pf2)
		
		M = np.zeros((self.size, self.size))
		M[np.ix_(pf, pc)] = 1
		P = np.zeros((self.size, self.size))
		P[np.ix_(pf1, pc1)] = 1
		P[np.ix_(pf2, pc2)] = 1
		self.Jp = self.J * (M - P)

	def generate_Hs(self):
		#sp = np.random.randint(0, 2, self.size) * 2 - 1
		self.hs = self.h# - np.dot(self.Jext, self.s - sp)

	def generate_Hp(self):
		sp = np.random.randint(0, 2, (self.size, self.size)) * 2 - 1
		snp = np.tile(self.s, (self.size, 1))
		self.hp = self.hs - np.sum(self.Jp * (snp - sp), axis=1)

	def dm(self, x, g, D): return 1 / np.sqrt(2 * np.pi) * \
            np.exp(-0.5 * x**2) * np.tanh(self.Beta*(g + x * np.sqrt(D)))

	def integrate_MFm(self, g, D):
		y, abserr = quad(self.dm, -np.inf, np.inf, args=(g, D))
		return y


	def fast_integrate_MFm(self, g, D):
		Nint = 50
		x = np.linspace(-3, 3, Nint)
		y = self.dm(x, g, D)
		y_int = cumtrapz(y, x)
		return y_int[-1]


	def integrate_main(self):
		self.h = self.H + np.dot(self.J,self.s)

		self.m = np.zeros(self.size)
		self.Z = np.zeros(self.size)
		
		g = self.h# - np.dot(self.Jext, self.s)  # mean of Hs
		D = np.zeros(self.size)#np.sum(self.Jext**2, axis=1)  # variance of Hs

		if self.integration_method == 'MC':
			for t in range(self.T):
				self.generate_Hs()
				self.m += np.tanh(self.Beta*self.hs) / self.T
		for i in range(self.size):		
			if D[i]==0:
					self.m[i] = np.tanh(self.Beta*g[i])
			else:
				if self.integration_method == 'MF':
					self.m[i] = self.integrate_MFm(g[i], D[i])
				if self.integration_method == 'FastMF':
					self.m[i] = self.fast_integrate_MFm(g[i], D[i])

	def integrate_partition(self):
		self.dJs = np.dot(self.Jp, self.s)
		self.Zp = np.zeros(self.size)
		self.mp = np.zeros(self.size)
		g = self.h# - np.dot(self.Jext, self.s)  # mean of Hs
		D = np.zeros(self.size)#np.sum(self.Jext**2, axis=1)  # variance of Hs
		gp = g - np.dot(self.Jp, self.s)  # mean of Hp
		Dp = D + np.sum(self.Jp**2, axis=1)  # variance of Hp
		
		if self.integration_method == 'MC':
			self.m = np.zeros(self.size)
			self.Z = np.zeros(self.size)
			for t in range(self.T):
				self.generate_Hs()
				self.m += np.tanh(self.Beta*self.hs) / self.T
				self.generate_Hp()
				self.mp += np.tanh(self.Beta*self.hp) / self.T
		else:
			for i in range(self.size):
				if Dp[i]==0:
					self.mp[i] = np.tanh(self.Beta*gp[i])
				else:
					if self.integration_method == 'MF':
						self.mp[i] = self.integrate_MFm(gp[i], Dp[i])
					if self.integration_method == 'FastMF':
						self.mp[i] = self.fast_integrate_MFm(gp[i], Dp[i])

	def EMD_distance(self):
		P1=0.5*(1+self.m)		# Probability of system elements being equal to 1
		P1p=0.5*(1+self.mp)		# Probability of partitioned system elements being equal to 1
		return np.abs(P1-P1p)	# EMD distance between distributions

	def phi(self):
		self.integrate_main()
		self.integrate_partition()
		phis = self.EMD_distance()
		phi = np.sum(phis[self.purview])
		return phi
		
	def phi_MIP(self,mechanism,purview):
	
		phiMIP=np.inf
		MIP=()
		
		self.set_mechanism_purview(mechanism,purview)
		self.integrate_main()
		count=0
		for npc1 in range(len(mechanism)):
			Mpartitions = list(combinations(self.mechanism, npc1))
			for cbp in Mpartitions:
				pc1 = list(cbp)
				pc2 = list(set(self.mechanism) - set(cbp))

				for npc2 in range(len(mechanism)):
					Fpartitions = list(combinations(self.purview, npc2))
					for fbp in Fpartitions:
						pf1 = list(fbp)
						pf2 = list(set(self.purview) - set(fbp))
						
						
						if min(len(pc1+pf1),len(pc2+pf2))>0:
							count+=1
							self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
							self.integrate_partition()
#							phis = self.m * self.dJs - self.Z + self.Zp
							phis = self.EMD_distance()
							phi = np.sum(phis[self.purview])
							if phiMIP>phi:
								phiMIP=phi
								MIP=(pc1, pc2, pf1, pf2)
#							if count%100==1:
#								print(pc1, pc2,pf1, pf2)
#								print(phi)
							
		print(count)
		print('MIP',MIP)
		print('phi',phiMIP)
		
		
	def phi_init_search_MIP(self,mechanism,purview,nM=None,nP=None):
	
		phiMIP=np.inf
		MIP=()
		
		self.set_mechanism_purview(mechanism,purview)
		self.integrate_main()
		pc1=[]; pc2=[]; pf1=[]; pf2=[]
		indicesM=list(range(len(self.mechanism)))
		indicesP=list(range(len(self.purview)))
		if nM is None:
			nM=np.random.randint(len(self.mechanism))
		if nP is None:
			nP=np.random.randint(len(self.purview))
		print(indicesM,indicesP,nM,nP)
		randomized_mechanism=np.random.permutation(self.mechanism)
		randomized_purview=np.random.permutation(self.purview)
		pc1=list(randomized_mechanism[indicesM[0:nM]])
		pc2=list(randomized_mechanism[indicesM[nM:]])
		pf1=list(randomized_purview[indicesP[0:nP]])
		pf2=list(randomized_purview[indicesP[nP:]])
		
		pc1, pc2,pf1, pf2 = correct_partition(pc1, pc2,pf1, pf2)

		print(pc1, pc2, pf1, pf2)
		self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
		self.integrate_partition()
		phis = self.EMD_distance()
		phi = np.sum(phis[self.purview])
		print(phi)
		self.phiMIP=phi
		self.MIP = (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
		
	def phi_search_MIP_step(self,mode=4):

		pc1, pc2,pf1, pf2 = copy_partition(self.MIP)
		P_prev= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
		pc1_p, pc2_p,pf1_p, pf2_p = copy_partition(self.MIP)
		
		if mode==4:
			mode=np.random.randint(4)
		
		change=False
		if mode == 0:
			for n in np.random.permutation(len(self.mechanism)):
#				print(mode,n,self.mechanism[n])
				pc1,pc2=change_partition_element(pc1.copy(),pc2.copy(),self.mechanism[n])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
				if valid_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()):
					self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
					self.integrate_partition()
					phis = self.EMD_distance()
					phi = np.sum(phis[self.purview])
	#				print((pc1, pc2,pf1, pf2),phi,self.phiMIP)
					if phi < self.phiMIP:
						print('change!',mode,self.mechanism[n],(pc1, pc2,pf1, pf2),phi,self.phiMIP)
						self.phiMIP=phi
						self.MIP = (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
						change=True
						break
					pc1,pc2,pf1,pf2=copy_partition(P_prev)

		elif mode == 1:
			for n in np.random.permutation(len(self.purview)):
#				print(mode,n,self.purview[n])
				pf1,pf2=change_partition_element(pf1.copy(),pf2.copy(),self.purview[n])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
				if valid_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()):
					self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
					self.integrate_partition()
					phis = self.EMD_distance()
					phi = np.sum(phis[self.purview])
	#				print((pc1, pc2,pf1, pf2),phi,self.phiMIP)
					if phi < self.phiMIP:
						print('change!',mode,self.purview[n],(pc1, pc2,pf1, pf2),phi,self.phiMIP)
						self.phiMIP=phi
						self.MIP= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
						change=True
						break

				pc1,pc2,pf1,pf2=copy_partition(P_prev)
				
		elif mode == 2:
			for n, m in ((u1, u2) for u1 in np.random.permutation(len(pc1_p)) for u2 in np.random.permutation(len(pc2_p))):
#				print(mode,(n,m),(pc1_p[n],pc2_p[m]))
				pc1,pc2=swap_partition_elements(pc1.copy(),pc2.copy(),pc1_p[n],pc2_p[m])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
				if valid_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()):
					self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
					self.integrate_partition()
					phis = self.EMD_distance()
					phi = np.sum(phis[self.purview])
	#				print((pc1, pc2,pf1, pf2),phi,self.phiMIP)
					if phi < self.phiMIP:
						print('change!',mode,(pc1_p[n],pc2_p[m]),(pc1, pc2,pf1, pf2),phi,self.phiMIP)
						self.phiMIP=phi
						self.MIP= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
						change=True
						break
					pc1,pc2,pf1,pf2=copy_partition(P_prev)
				
		elif mode == 3:
			for n, m in ((u1, u2) for u1 in np.random.permutation(len(pf1_p)) for u2 in np.random.permutation(len(pf2_p))):
#				print(mode,(n,m),(pf1_p[n],pf2_p[m]))
				pf1,pf2=swap_partition_elements(pf1.copy(),pf2.copy(),pf1_p[n],pf2_p[m])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
				if valid_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()):
					self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
					self.integrate_partition()
					phis = self.EMD_distance()
					phi = np.sum(phis[self.purview])
#					print('test',(pc1, pc2,pf1, pf2),phi,self.phiMIP)
					if phi < self.phiMIP:
						print('change!',mode,(pf1_p[n],pf2_p[m]),(pc1, pc2,pf1, pf2),phi,self.phiMIP)
						self.phiMIP=phi
						self.MIP= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
						change=True
						break
					pc1,pc2,pf1,pf2=copy_partition(P_prev)
				
		return change

	def search_MIP(self,mechanism,purview,nM=None,nP=None):
		self.phi_init_search_MIP(mechanism,purview,nM,nP)
		change=True
		while change:
			change1=False
			for n in np.random.permutation(4):
				change2=self.phi_search_MIP_step(n)
				change1 = change1 or change2
				print('permut',n,change2,change1, ' - ',self.MIP)
			change = change1
		print('found!',self.MIP,self.phiMIP)
			
			
			
	def phi_SA_MIP_step(self,beta):

		pc1, pc2,pf1, pf2 = copy_partition(self.MIP)
		
		mode=np.random.randint(4)
		
		if mode == 0 and len(self.mechanism):
			n=np.random.randint(len(self.mechanism))
			pc1,pc2=change_partition_element(pc1.copy(),pc2.copy(),self.mechanism[n])
		elif mode == 1 and len(self.purview):
			n=np.random.randint(len(self.purview))
			pf1,pf2=change_partition_element(pf1.copy(),pf2.copy(),self.purview[n])
		elif mode == 2 and len(pc1) and len(pc2):
			n=np.random.randint(len(pc1))
			m=np.random.randint(len(pc2))
			pc1,pc2=swap_partition_elements(pc1.copy(),pc2.copy(),pc1[n],pc2[m])
		elif mode == 3 and len(pf1) and len(pf2):
			n=np.random.randint(len(pf1))
			m=np.random.randint(len(pf2))
			pf1,pf2=swap_partition_elements(pf1.copy(),pf2.copy(),pf1[n],pf2[m])
		
		
		if valid_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()):
			self.set_partition(pc1.copy(), pc2.copy(), pf1.copy(), pf2.copy())
			self.integrate_partition()
			phis = self.EMD_distance()
			phi = np.sum(phis[self.purview])
#			print(-beta,phi - self.phiMIP)
			
			if phi < self.phiMIP or np.log(np.random.rand()) < -beta*(phi -  self.phiMIP):
				print(phi,self.phiMIP,beta,(phi - self.phiMIP))
				print('change!',mode,(pc1, pc2,pf1, pf2),phi,self.phiMIP)
				self.phiMIP=phi
				self.MIP = (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#			if phi < self.phiMIP:
#				self.phiMIP=phi			
		

			
#	def phi_search_MIP(self,mechanism,purview):
#	
#		phiMIP=np.inf
#		MIP=()
#		
#		self.set_mechanism_purview(mechanism,purview)
#		self.integrate_main()
#		pc1=[]; pc2=[]; pf1=[]; pf2=[]
#		indicesM=list(range(len(self.mechanism)))
#		nM=np.random.randint(len(self.mechanism))
#		indicesP=list(range(len(self.purview)))
#		nP=np.random.randint(len(self.purview))
#		print(indicesM,indicesP,nM,nP)
#		pc1=list(np.array(self.mechanism)[indicesM[0:nM]])
#		pc2=list(np.array(self.mechanism)[indicesM[nM:]])
#		pf1=list(np.array(self.purview)[indicesP[0:nP]])
#		pf2=list(np.array(self.purview)[indicesP[nP:]])
#		
#		print(pc1, pc2, pf1, pf2)
##		for n in mechanism:
##			if np.random.randint(2)==0:
##				pc1+=[n] 
##			else: 
##				pc2+=[n]
##		for n in purview:
##			if np.random.randint(2)==0: 
##				pf1+=[n] 
##			else: 
##				pf2+=[n]
#			
#		pc1, pc2,pf1, pf2 = correct_partition(pc1, pc2,pf1, pf2)
#		
#		
#		P_prev= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#		
#		print(pc1, pc2, pf1, pf2)
#		self.set_partition(pc1, pc2, pf1, pf2)
#		self.integrate_partition()
#		phis = self.EMD_distance()
#		phi = np.sum(phis[self.purview])
#		print(phi)
#		phiMIP=phi
#		
#		change=True
#		while(change):
#			change=False
#			for n in np.random.permutation(range(len(self.mechanism))):
#				pc1,pc2=change_partition_element(pc1.copy(),pc2.copy(),self.mechanism[n])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#				self.set_partition(pc1, pc2, pf1, pf2)
#				self.integrate_partition()
#				phis = self.EMD_distance()
#				phi = np.sum(phis[self.purview])
##				print('c',mechanism[n],phi,(pc1, pc2,pf1, pf2))
#				if phi < phiMIP:
#					phiMIP=phi
#					P_prev= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#					change=True
##					print('change!')
#				pc1,pc2,pf1,pf2=P_prev
#			
#			for n in np.random.permutation(range(len(self.purview))):
#				pf1,pf2=change_partition_element(pf1.copy(),pf2.copy(),self.purview[n])
#				pc1, pc2,pf1, pf2 = correct_partition(pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#				self.set_partition(pc1, pc2, pf1, pf2)
#				self.integrate_partition()
#				phis = self.EMD_distance()
#				phi = np.sum(phis[self.purview])
##				print('f',purview[n],phi,(pc1, pc2,pf1, pf2))
#			
#				if phi < phiMIP:
#					phiMIP=phi
#					P_prev= (pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy())
#					change=True
##					print('change!')
#				pc1, pc2,pf1, pf2=P_prev
#				
#				
#				
#			
#			print(P_prev, phiMIP,change)
#			
	
def copy_partition(partition):
	return partition[0].copy(),partition[1].copy(),partition[2].copy(),partition[3].copy()
	
def correct_partition(pc1, pc2,pf1, pf2):
	if len(pc1+pf1)==0 or len(pc2+pf2)==0:
		if np.random.randint(2)==0:
			pc=pc1+pc2
			pc1,pc2=change_partition_element(pc1.copy(),pc2.copy(),pc[np.random.randint(len(pc))])
		else:
			pf=pf1+pf2
			pf1,pf2=change_partition_element(pf1.copy(),pf2.copy(),pf[np.random.randint(len(pf))])
	return pc1.copy(), pc2.copy(),pf1.copy(), pf2.copy()
	
def valid_partition(pc1, pc2,pf1, pf2):
	return len(pc1+pf1)>0 and len(pc2+pf2)>0
	
	
def swap_partition_elements(p1,p2,n,m):
	if n in p1:
		p1.remove(n)
		p2.remove(m)
		p1+=[m]
		p2+=[n]
	elif n in p2:
		p2.remove(n)
		p1.remove(m)
		p1+=[n]
		p2+=[m]
	return p1.copy(),p2.copy()
	
def change_partition_element(p1,p2,n):
	if n in p1:
		p1.remove(n)
		p2+=[n]
	elif n in p2:
		p2.remove(n)
		p1+=[n]
	return p1.copy(),p2.copy()

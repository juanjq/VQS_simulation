import numpy             as np
import hamiltonian       as ham
from   quspin.basis      import spin_basis_1d  


########################################################
# measure
########################################################
        
# a function to simulate a quantum probabilistic measure
def measure(H,psi):
    
    E,V = H.eigh()
    V   = np.transpose(V)
    
    Coef    = [np.vdot(V[i],psi) for i in range(H.basis.Ns)]   # coeficients of state
    Prob    = [abs(Coef[i])**2   for i in range(len(Coef)) ]   # probabilities
    IntProb = [sum(Prob[:i+1]) for i in range(len(Prob))]      # probabilities "integration"

    #random probability assignment
    DiceRoll = np.random.rand()*sum(Prob)
    for i in range(len(Prob)):
        if i==0:
            if (DiceRoll >= 0) and (DiceRoll <= IntProb[i]):
                index=i
        else:
            if (DiceRoll >= IntProb[i-1]) and ((DiceRoll <= IntProb[i])):
                index=i

    return E[index]


def multiMeasure_test(H_array,psi,Nmeasure):
    
    measure_array = []
    
    for i in range(len(H_array)):
        measure_array.append([measure(H_array[i],psi) for j in range(Nmeasure)])
    
    
    E_array = []
    for i in range(len(np.transpose(measure_array))):
        E_array.append(sum(np.transpose(measure_array)[i]))
        
    E_mean  = np.mean(np.mean(E_array))
    
    return E_mean

def exact_sol(N):
    
    
    H=ham.XXZ(N,1,1,1,1,1)

    E,V=H.eigsh(k=2,which="SA",maxiter=1E4)
    V=np.transpose(V)[0]
    
    if E[0]<=E[1]:
        E0=E[0]
        E1=E[1]
    else:
        E0=E[1]
        E1=E[0]
    
    return V,E0,E1

########################################################

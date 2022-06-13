import numpy       as np
import hamiltonian as ham
import auxiliar    as aux
import matrices    as mat
from   scipy       import misc
import random
import time


#parameter definition
def parameters(NUM,NCYCLES,ANSATZ,MODE,NMEASURE,PATH):
    global N
    N = NUM
    global ncycles
    ncycles = NCYCLES
    global ansatz
    ansatz = ANSATZ
    global mode
    mode = MODE
    global Nmeasure
    Nmeasure = NMEASURE
    global path
    path = PATH

########################################################
# quantum evolution
########################################################

# the full evolve function chosing the mode
def evolve(theta,ncycles,mode):  

    N = int(np.log2(len(ansatz)))
    
    if mode == 'evenodd':
        U = mat.matrix_evenodd(N,theta,ncycles)
        
    elif mode == 'pairs':
        U = mat.matrix_pairs(N,theta,ncycles)
        
    elif mode == 'pairs_nnn':
        U = mat.matrix_pairs_nnn(N,theta,ncycles)
        
    elif mode == 'pairs_conc':
        U = mat.matrix_pairs_conc(N,theta,ncycles)
               
    elif mode == 'individual':
        U = mat.matrix_indiv(N,theta,ncycles)
        
    return np.dot(U,ansatz)

########################################################


########################################################
# measure
########################################################

# exact measuring
def multiMeasure(H_array,psi,Nmeasure):
        
    H = sum(H_array)
    E_exact = np.real(aux.braket(H.toarray(),psi))
    
    return E_exact

########################################################


########################################################
# RED random estimation descent
########################################################

def nextStep(E,T,d):

    dt= (T[0]-T[2])*abs((E[2]-E[0])/E[0])+(T[0]-T[1])*abs((E[1]-E[0])/E[0])

    v,EE,E1=exact_sol(N)
    theta= T[0]+np.random.choice([1,-1])*aux.normalize(dt)*d*abs((E[1]-EE*1.1)/EE)

    return theta

def nextStepRand(E,T,d):

    theta= T[0]+aux.normalize([random.gauss(0,1) for i in range(len(T[0]))])*d

    return theta

def randomize(v,d=0):
    return aux.normalize([v[i]+v[i]*(random.gauss(0,1))*d for i in range(len(list(v)))])


def step(E,T,ansatz,ncycles,Hdec,alpha,iteration,t,Vex,rnd_mode='0'):

    if rnd_mode == '0':
        theta = nextStep(E,T,d=alpha[0])
    elif rnd_mode == '1':
        theta = nextStepRand(E,T,d=alpha[1])
    elif rnd_mode == '2':
        theta = nextStepRand(E,T,d=alpha[2])
    elif rnd_mode == '3':
        theta = nextStepRand(E,T,d=alpha[3])

    Ebefore = E[0]
    Etheta, psi  = energy(theta,vec=True)
    
    T.append(theta)
    E.append(Etheta)


    E,T = aux.sort(E,T)
    E = E[:-1]
    T = T[:-1] 
    
    if Ebefore > E[0]:
        aux.write(str(iteration)+','+str(E[0])+','+str(aux.fidelity(Vex,psi))+','+str(t),'random_estimation_descent',path)
        
    return E,T,Ebefore

def descent_test(t0,nb_max_iter,counter,alpha,total_t):
    
    start_time = time.time()
    def tim():
        return time.time() - start_time

    num  = aux.nVariables(N,ncycles,mode)
    H_array = [ham.XXZ_X(N),ham.XXZ_Y(N),ham.XXZ_Z(N)]

    # thetas
    init0 = t0  
    init1 = init0 + randomize([0.6 for i in range(len(init0))],0.5)
    init2 = init0 + randomize([0.6 for i in range(len(init0))],0.5)

    E0 = energy(init0)
    E1 = energy(init1)
    E2 = energy(init2)

    # initial vectors
    Einit = [E0,E1,E2]
    Tinit = [init0,init1,init2]

    E,T = aux.sort(Einit,Tinit)
    theta = T[0]
    
    Vex,E0ex,E1ex = exact_sol(N)
    Eex, psi = energy(T[0],vec=True)
    aux.write(str(0)+','+str(Einit[0])+','+str(aux.fidelity(Vex,psi))+','+str(tim()),'random_estimation_descent',path)

    repetitionCounter, Ebefore = 0, 0
    for iteration in range(nb_max_iter):
        
        if tim()<total_t:
            
            if Ebefore == E[0]:
                repetitionCounter = repetitionCounter+1   
            else:
                repetitionCounter = 0

            if (repetitionCounter<random.gauss(counter[0],20)):

                E,T,Ebefore = step(E,T,ansatz,ncycles,H_array,alpha,iteration+1,tim(),Vex,rnd_mode='0')

            elif (repetitionCounter<random.gauss(counter[1],20)):

                E,T,Ebefore = step(E,T,ansatz,ncycles,H_array,alpha,iteration+1,tim(),Vex,rnd_mode='1') 

            elif (repetitionCounter<random.gauss(counter[2],10)):

                E,T,Ebefore = step(E,T,ansatz,ncycles,H_array,alpha,iteration+1,tim(),Vex,rnd_mode='2') 

            elif (repetitionCounter<random.gauss(counter[3],5)):

                E,T,Ebefore = step(E,T,ansatz,ncycles,H_array,alpha,iteration+1,tim(),Vex,rnd_mode='3') 

            else:
                break
        else:
            break

########################################################


########################################################
# descent gradient
########################################################

def partial_derivative(func, var=0, point=[]):
    
    args = point[:]
    
    def wraps(x):
        
        args[var] = x
        
        return func(args)
    
    return misc.derivative(wraps, point[var], dx = 1e-6)

def gradient_descent(f,v0,alpha,eps,nb_max_iter,total_t,file='descent_gradient',write_mode=False):
    
    start_time = time.time()
    def tim():
        return time.time() - start_time
    
    V=v0

    E0,psi = f(V,vec=True)

    cond = eps + 10
    nb_iter = 0 
    tmp_E0 = E0
    
    Vex,E1ex,E2ex = exact_sol(N)
    
    if write_mode == True:
        aux.write(str(nb_iter)+','+str(E0)+','+str(aux.fidelity(Vex,psi))+','+str(tim())+','+str(mode),file,path)
    else:
        aux.write(str(nb_iter)+','+str(E0)+','+str(aux.fidelity(Vex,psi))+','+str(tim()),file,path)
    
    while  nb_iter < nb_max_iter and tim()<total_t:
        
        tmp_V = [ V[i] - alpha * partial_derivative(f,i,V) for i in range(len(V))]
        V = tmp_V

        E0, psi = f(V,vec=True)

        nb_iter = nb_iter + 1
        cond = abs( tmp_E0 - E0 )
        
        tmp_E0 = E0
        
        if write_mode == True:
            aux.write(str(nb_iter)+','+str(E0)+','+str(aux.fidelity(Vex,psi))+','+str(tim())+','+str(mode),file,path)
        else:
            aux.write(str(nb_iter)+','+str(E0)+','+str(aux.fidelity(Vex,psi))+','+str(tim()),file,path)


def energy(theta,vec=False):
    
    psi=evolve(theta,ncycles,mode)
    
    H_array = [ham.XXZ_X(N),ham.XXZ_Y(N),ham.XXZ_Z(N)]
    E0 = multiMeasure(H_array,psi,Nmeasure)
    
    if vec == False:
        return E0
    else:
        return E0, psi

########################################################

########################################################
# exact solution
########################################################

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

########################################################
# ansatz
########################################################

def psi_ansatz(N):
    
    # selected basis elements that lie on the momentum block
    i3 = [3,5,6]
    i4 = [3,5,6,7,9,10,11,12,13,14]
    i5 = [7,11,13,14,19,21,22,25,26,28]
    i6 = [15,23,27,29,30,39,43,45,46,51,53,54,57,58,60]
    i7 = [15,23,27,29,30,39,43,45,46,51,53,54,57,58,60,71,75,77,78,83,85,86,89,90,92,99,101,102,105,106,108,113,114,116,120]
    i8 = [31,47,55,59,61,62,79,87,91,93,94,103,107,109,110,115,117,118,121,122,124,143,151,155,157,158,167,171,173,174,
         179,181,182,185,186,188,199,203,205,206,211,213,214,217,218,220,227,229,230,233,234,236,241,242,244,248]
    indexes = [None,None,None,i3,i4,i5,i6,i7,i8,i9]
    
    # empty array as the vector ansatz
    psi = [0 for i in range(2**N)]
    
    # selecting the indexes correspondent to each basis
    for i in indexes[N]:
        psi[i]=1  

    return aux.normalize(psi)


def theta_ansatz(N,L,mode,ncycles,seed=None):
    
    num = aux.nVariables(N,ncycles,mode)
    
    if seed != None:
        np.random.seed(seed)
    
    t = np.random.rand(num)*2*L-L
    
    return t

########################################################
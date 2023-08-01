# VQS simulations
Simulation in python of VQS methods. We will introduce the different functions implemented for our calculus.

<p align="center">
  <img src="https://github.com/juanjq/VQS_simulation/blob/main/graphs/MainScheme.png" width=470" height="500">
</p>

* For taking the data of measurements use file `measures/data_measuring.ipynb`.
* For measurement plots `measures/plot_script.ipynb`

* For the DG or RED data use the file `results/data_results.ipynb`.
* For DG/RED plots `results/plot_script.ipynb`

## DG algorithm 
Implementation of *descent gradient* algorithm.

```
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
    if write_mode == True:    aux.write(str(nb_iter)+','+str(E0)+','+str(aux.fidelity(Vex,psi))+','+str(tim())+','+str(mode),file,path)
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
```
With the cost function,
```
def energy(theta,vec=False):
    psi=evolve(theta,ncycles,mode)
    H_array = [ham.XXZ_X(N),ham.XXZ_Y(N),ham.XXZ_Z(N)]
    E0 = multiMeasure(H_array,psi,Nmeasure)
    if vec == False:
        return E0
    else:
        return E0, psi
```

## RED algorithm 
Implementation of *random estimation descent* algorithm.

<p align="center">
  <img src="https://github.com/juanjq/VQS_simulation/blob/main/graphs/algorithm-1.png" width="400" height="500">
</p>

The main function,

```
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
```
And the subfunctions,
```
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
```

<p align="center">
  <img src="https://github.com/juanjq/VQS_simulation/blob/main/graphs/opt_gif.gif" width="500" height="430">
</p>

## Multimeasure algorithm
Implementation classically of the quantum probabilistic measurement.

```
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
```
And multimeasures
```
def multiMeasure_test(H_array,psi,Nmeasure):
    measure_array = []
    for i in range(len(H_array)):
        measure_array.append([measure(H_array[i],psi) for j in range(Nmeasure)])
    E_array = []
    for i in range(len(np.transpose(measure_array))):
        E_array.append(sum(np.transpose(measure_array)[i]))    
    E_mean  = np.mean(np.mean(E_array))
    return E_mean
```

## Quantum circuit simulation
Implementation of the quantum circuit.

<p align="center">
  <img src="https://github.com/juanjq/VQS_simulation/blob/main/graphs/Circuits.png" width="670" height="430">
</p>

Evolution described with,

```
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
```

And the matrices definition at:

**Even-Odd staggering**

```
def matrix_evenodd(N,theta,ncycles):
    nn=3
    # hamiltonians of J1 and J2    
    hXX = ham.XXZ(N,1,1,0,0,1).toarray()
    hZodd  = sum([ham.Z(N,i) for i in range(N) if i%2==0 ]).toarray()
    hZeven = sum([ham.Z(N,i) for i in range(N) if i%2!=0 ]).toarray() 
    # we define the number of cycles we use
    def Ucycle(n):
        # first elements J1 and J2
        U_array = [ expm(-1j*hZeven*theta[0+n*nn]),expm(-1j*hZodd*theta[1+n*nn]),expm(-1j*hXX*theta[2+n*nn])]
        return U_array
    # multiplying the total matrix
    U = expm(np.zeros((2**N,2**N)))
    for i in range(ncycles):  
        cycle=Ucycle(i)
        for j in range(len(cycle)):
            U=np.matmul(U,cycle[j])      
    return U
```
**Pairs staggering**
```
def matrix_pairs(N,theta,ncycles):
    nn = int(N/2-N%2/2)
    NN = 2*nn+1
    # hamiltonians of J1 and J2    
    hXX = ham.XXZ(N,1,1,0,0,1).toarray()
    h_pairs_left=[]
    for i in range(nn):
        h_pairs_left.append((ham.Z(N,2*i)+ham.Z(N,2*i+1)).toarray()) 
    h_pairs_right=[]
    for i in range(nn):
        h_pairs_right.append((ham.Z(N,N-1-(2*i))+ham.Z(N,N-1-(2*i+1))).toarray())           
    def Ucycle(n):
        # first elements
        U_array = []
        jj = 0
        while jj < len(h_pairs_left):
            U_array.append(expm(-1j*h_pairs_left[jj]*theta[jj+n*NN]))
            jj = jj+1 
        kk = 0    
        while kk < len(h_pairs_right):
            U_array.append(expm(-1j*h_pairs_right[kk]*theta[jj+n*NN]))
            jj = jj+1
            kk = kk+1
        U_array.append(expm(-1j*hXX*theta[jj+n*NN]))                  
        return U_array
    # multiplying the total matrix
    U = expm(np.zeros((2**N,2**N)))
    for i in range(ncycles):
        cycle=Ucycle(i)
        for j in range(len(cycle)):
            U=np.matmul(U,cycle[j])  
    return U
```
**Pairs-NNN staggering**
```
def matrix_pairs_nnn(N,theta,ncycles):
    nn = int(N/2-N%2/2)
    NN = 2*nn+1
    # hamiltonians of J1 and J2    
    hXX = ham.XXZ(N,1,1,0,0,1).toarray()
    h_pairs_left=[]
    for i in range(nn):
        if i%2==0:
            h_pairs_left.append((ham.Z(N,2*i)+ham.Z(N,2*i+2)).toarray())
        else:
            h_pairs_left.append((ham.Z(N,2*(i-1)+1)+ham.Z(N,2*(i-1)+1+2)).toarray())  
    h_pairs_right=[]
    for i in range(nn):
        if i%2==0:
            h_pairs_right.append((ham.Z(N,N-1-(2*i))+ham.Z(N,N-1-(2*i+2))).toarray())
        else:
            h_pairs_right.append((ham.Z(N,N-1-(2*(i-1)+1))+ham.Z(N,N-1-(2*(i-1)+1+2))).toarray())     
    def Ucycle(n):
        # first elements
        U_array = []
        jj = 0
        while jj < len(h_pairs_left):
            U_array.append(expm(-1j*h_pairs_left[jj]*theta[jj+n*NN]))
            jj = jj+1   
        kk = 0    
        while kk < len(h_pairs_right):
            U_array.append(expm(-1j*h_pairs_right[kk]*theta[jj+n*NN]))
            jj = jj+1
            kk = kk+1
        U_array.append(expm(-1j*hXX*theta[jj+n*NN]))                  
        return U_array
    # multiplying the total matrix
    U = expm(np.zeros((2**N,2**N)))
    for i in range(ncycles): 
        cycle=Ucycle(i)
        for j in range(len(cycle)):
            U=np.matmul(U,cycle[j])  
    return U
```
**Individual staggering**
```
def matrix_indiv(N,theta,ncycles):
    nn = N
    NN = nn+1
    # hamiltonians of J1 and J2    
    hXX = ham.XXZ(N,1,1,0,0,1).toarray()
    h=[]
    for i in range(nn):
            h.append(ham.Z(N,i).toarray())
    # we define the number of cycles we use
    def Ucycle(n):   
        # first elements
        U_array = []
        jj=0
        while jj < len(h):
            U_array.append(expm(-1j*h[jj]*theta[jj+n*NN]))
            jj=jj+1
        U_array.append(expm(-1j*hXX*theta[jj+n*NN]))                    
        return U_array
    # multiplying the total matrix
    U = expm(np.zeros((2**N,2**N)))
    for i in range(ncycles):
        cycle=Ucycle(i)
        for j in range(len(cycle)):
            U=np.matmul(U,cycle[j])  
    return U
```
## Hamiltonian definition
We create a script with all the hamiltonians needed using QuSpin.

<p align="center">
  <img src="https://github.com/juanjq/VQS_simulation/blob/main/graphs/hamiltonian.png" width="400" height="500">
</p>

```
def XXZ(N,j1,j2,d1,d2,h): 
    basis = spin_basis_1d(N,pauli=False)
    J1 = [[j1,i,i+1] for i in range(N-1)]
    J2 = [[j2,i,i+2] for i in range(N-2)]    
    D1 = [[d1,i,i+1] for i in range(N-1)]
    D2 = [[d2,i,i+2] for i in range(N-2)]
    HT = [[h,i     ] for i in range(N)  ]
    static = [["xx",J1],["yy",J1],["xx",J2],["yy",J2],["zz",D1],["zz",D2],["z",HT]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)


def XXZ_X(N):
    basis = spin_basis_1d(N,pauli=False)
    J1 = [[1,i,i+1] for i in range(N-1)]
    J2 = [[1,i,i+2] for i in range(N-2)]    
    static = [["xx",J1],["xx",J2]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)

def XXZ_Y(N):
    basis = spin_basis_1d(N,pauli=False)
    J1 = [[1,i,i+1] for i in range(N-1)]
    J2 = [[1,i,i+2] for i in range(N-2)]    
    static = [["yy",J1],["yy",J2]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)
    
def XXZ_Z(N):
    basis = spin_basis_1d(N,pauli=False)
    D1 = [[1,i,i+1] for i in range(N-1)]
    D2 = [[1,i,i+2] for i in range(N-2)]    
    HT = [[1,i    ] for i in range(N)  ]
    static = [["zz",D1],["zz",D2],["z",HT]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)

def ZZ(N,n,m):
    basis = spin_basis_1d(N,pauli=False)
    Z=[[1,i] for i in range(N) if ((i==n) or (i==m))]
    static = [["z",Z]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)  

def Z(N,n):
    basis = spin_basis_1d(N,pauli=False)
    Z=[[1,i] for i in range(N) if (i==n)]
    static = [["z",Z]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)  

def Zeven(N): 
    basis = spin_basis_1d(N,pauli=False) 
    Z=[[1,i] for i in range(N) if (i%2==0)]
    static = [["z",Z]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False) 

def Zodd(N):
    basis = spin_basis_1d(N,pauli=False)
    Z=[[1,i] for i in range(N) if (i%2!=0)]
    static = [["z",Z]] 
    return hamiltonian(static,[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False)  

def zeros(N):
    basis = spin_basis_1d(N,pauli=False)
    return hamiltonian([],[],basis=basis,dtype=np.float64,check_herm=False,check_symm=False) 
```

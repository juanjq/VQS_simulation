# VQS simulations
Simulation in python of VQS methods. We will introduce the different functions implemented for our calculus.

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

## Multimeasure algorithm
Implementation classically of the quantum probabilistic measurement.

## Quantum circuit simulation
Implementation of the quantum circuit.

## Data generation
All methods toether to extract data.

## Hamiltonian definition
We create a script with all the hamiltonians needed.

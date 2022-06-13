import numpy       as np
import hamiltonian as ham
from scipy.linalg  import expm


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


def matrix_pairs_conc(N,theta,ncycles):
    
    nn = N-1
    NN = nn+1
    
    # hamiltonians of J1 and J2    
    hXX = ham.XXZ(N,1,1,0,0,1).toarray()

    h_pairs=[]
    for i in range(nn):
            h_pairs.append((ham.Z(N,i)+ham.Z(N,i+1)).toarray())

    # we define the number of cycles we use
    def Ucycle(n):
        
        # first elements
        U_array = []
        jj=0
        while jj < len(h_pairs):
            U_array.append(expm(-1j*h_pairs[jj]*theta[jj+n*NN]))
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
#QuSpin basics
from quspin.operators import hamiltonian   # Hamiltonians and operators 
from quspin.basis     import spin_basis_1d # Hilbert space spin basis     

import numpy as np


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
 

import numpy as np
from matplotlib import pyplot as plt

def mc():
    # incoming hydrograph
    t=np.arange(0,11,1)*12.*60
    I=np.array([56.7,56.7,85,113,142,170,142,113,85,56.7,56.7])

    # Muskingum-Cunge calcaultion
    dt=180.                             # time step
    tf=150*60                           # final time
    M=int(tf/dt)                        # number of time steps
    tv=np.linspace(t[0],tf,M+1)         # time vector
    dt=tv[1]-tv[0]                      # recalculate time step
    L=4600.                             # total reach length
    N=20                                # number of subreaches
    dx=L/N                              # subreach length
    def cf(Q,n,S0,B):                   # celerity function
        return 5*S0**0.5/(3*n)*(Q*n/(B*S0**0.5))**0.4
    qp=0.5*(np.max(I)+I[0])             # average flow
    # qp=I[0]                           # initial flow
    # qp=np.max(I)                      # peak flow
    B=60.                               # channel width
    n=0.0333                            # Manning's roughness
    S0=0.01                             # channel slope
    c=cf(qp,n,S0,B)                     # celerity
    K=dx/c                              # K parameter
    X=0.5*(1-qp/(B*c*S0*dx))            # X parameter
    D=2*K*(1-X)+dt                      # MC denominator
    C1=(dt-2*K*X)/D                     # MC coefficient 1
    C2=(dt+2*K*X)/D                     # MC coefficient 2
    C3=(2*K*(1-X)-dt)/D                 # MC coefficient 3
    Q=np.zeros((N,M+1))                 # outflow matrix
    Q[0,:]=np.interp(tv, t, I)          # set inflow boundary condition
    Q[:,0]=Q[0,0]                       # set initial condition
    for i in range(1,N):                # for each subreach
        for j in range(1,M+1):              # for each timestep
            Q[i,j]=C1*Q[i-1,j]+C2*Q[i-1,j-1]+C3*Q[i,j-1]    # MC calc

    f,ax=plt.subplots(1,1)
    for i in range(N):                  # plot subreach snapshots
        ax.plot(tv/60, Q[i,:], '-')

    ax.set_xlim([0,160])
    ax.set_xlabel('time [min]')
    ax.set_ylabel('flow [m^3/s]')
    ax.set_title('C1={:3.2f}, C2={:3.2f}, C3={:3.2f}'.format(C1,C2,C3))
    plt.show()

def main():
    mc()

if __name__=="__main__":
    main()
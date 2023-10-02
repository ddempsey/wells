import numpy as np

def power_method():
    A=np.array([[3,4],[2,1]])
    x=np.array([1,1]).T

    for i in range(3):
        print(i)
        Ax=np.dot(A,x)
        print(Ax)
        x=Ax/np.sqrt(np.dot(Ax,Ax))
        print(x)
        ev=np.dot(x.T, np.dot(A,x))
        print(ev)

def divergence():
    def u(x,y,z): return np.array([3*np.sin(x)*z,2*x*y,-y*z**2])
    def divu(x,y,z): return 3*np.cos(x)*z+2*x+-2*y*z
    
    dx,dy,dz=[0.1,0.1,0.01]
    x0,y0,z0=[1,2,3]

    dudx=(u(x0+dx,y0,z0)-u(x0-dx,y0,z0))[0]/(2*dx)
    dudy=(u(x0,y0+dy,z0)-u(x0,y0-dy,z0))[1]/(2*dy)
    dudz=(u(x0,y0,z0+dz)-u(x0,y0,z0-dz))[2]/(2*dz)

    print('analytical:', divu(x0,y0,z0))
    print('numerical:', dudx+dudy+dudz)

    print(3*np.sin(1.1)*3)

def main():
    #power_method()
    divergence()

if __name__=="__main__":
    main()
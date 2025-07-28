# -*- coding: utf-8 -*-
import torch
import numpy as np
from qiskit import QuantumCircuit,QuantumRegister,transpile
from qiskit_aer import Aer
from scipy.fft import fft2, ifft2

def qkm_true(input,p,net):
    '''
    Args:
        input: torch.tensor (3,h,h)
        p: int in [0,t-1]
    Returns:
        np.array -> net(input,p) (h,h)
        list of QuantumCircuit -> qc0,qc1,qc2,qc3,qc4
    '''
    net.eval()
    backend = Aer.get_backend('statevector_simulator') # 选择模拟器
    
    r11,r23,r32,r42,r52 = net.encr(input)
    phi11,phi23,phi32,phi42,phi52 = net.encphi(input)
    
    # r11,phi11
    b,c,h,w = phi11.shape
    assert b==1 and int(c*h*w)==net.kno1.kno.features
    features = net.kno1.kno.features
    
    phi11 = phi11.view(features).detach().numpy()
    
    counts :int = int(np.log2(features))
    q0 = QuantumRegister(counts,name='q0')
    qc0 = QuantumCircuit(q0)
    qc0.initialize(np.exp(1.j*np.asarray(phi11,dtype=np.complex128))/(np.power(2,counts/2,dtype=np.float64)), q0)
    for i in range(counts):
        qc0.rz(net.kno1.kno.weight.detach().numpy()[i] * p , q0[counts-1-i])
    job = backend.run(transpile(qc0,backend))
    state = job.result().get_statevector(qc0) # get_statevector 对应于 statevector_simulator
    phi11 = torch.from_numpy(np.array(state).astype(np.complex64)).angle()
    phi11 = phi11.view(b,c,h,w)
    x11 = r11 * torch.cos(phi11)

    # r23,phi23
    b,c,h,w = phi23.shape
    assert b==1 and int(c*h*w)==net.kno2.kno.features
    features = net.kno2.kno.features
    phi23 = phi23.view(features).detach().numpy()
    counts = int(np.log2(features))
    q1 = QuantumRegister(counts,name='q1')
    qc1 = QuantumCircuit(q1)
    qc1.initialize(np.exp(1.j*np.asarray(phi23,dtype=np.complex128))/(np.power(2,counts/2,dtype=np.float64)), q1)
    for i in range(counts):
        qc1.rz(net.kno2.kno.weight.detach().numpy()[i] * p , q1[counts-1-i])
    job = backend.run(transpile(qc1,backend))
    state = job.result().get_statevector(qc1) # get_statevector 对应于 statevector_simulator
    phi23 = torch.from_numpy(np.array(state).astype(np.complex64)).angle()
    phi23 = phi23.view(b,c,h,w)
    x23 = r23 * torch.cos(phi23)
    
    # r32,phi32
    b,c,h,w = phi32.shape
    assert b==1 and int(c*h*w)==net.kno3.kno.features
    features = net.kno3.kno.features
    phi32 = phi32.view(features).detach().numpy()
    counts = int(np.log2(features))
    q2 = QuantumRegister(counts,name='q2')
    qc2 = QuantumCircuit(q2)
    qc2.initialize(np.exp(1.j*np.asarray(phi32,dtype=np.complex128))/(np.power(2,counts/2,dtype=np.float64)), q2)
    for i in range(counts):
        qc2.rz(net.kno3.kno.weight.detach().numpy()[i] * p , q2[counts-1-i])
    job = backend.run(transpile(qc2,backend))
    state = job.result().get_statevector(qc2) # get_statevector 对应于 statevector_simulator
    phi32 = torch.from_numpy(np.array(state).astype(np.complex64)).angle()
    phi32 = phi32.view(b,c,h,w)
    x32 = r32 * torch.cos(phi32)
    
    # r42,phi42
    b,c,h,w = phi42.shape
    assert b==1 and int(c*h*w)==net.kno4.kno.features
    features = net.kno4.kno.features
    phi42 = phi42.view(features).detach().numpy()
    counts = int(np.log2(features))
    q3 = QuantumRegister(counts,name='q3')
    qc3 = QuantumCircuit(q3)
    qc3.initialize(np.exp(1.j*np.asarray(phi42,dtype=np.complex128))/(np.power(2,counts/2,dtype=np.float64)), q3)
    for i in range(counts):
        qc3.rz(net.kno4.kno.weight.detach().numpy()[i] * p , q3[counts-1-i])
    job = backend.run(transpile(qc3,backend))
    state = job.result().get_statevector(qc3) # get_statevector 对应于 statevector_simulator
    phi42 = torch.from_numpy(np.array(state).astype(np.complex64)).angle()
    phi42 = phi42.view(b,c,h,w)
    x42 = r42 * torch.cos(phi42)
    
    # r52,phi52
    b,c,h,w = phi52.shape
    assert b==1 and int(c*h*w)==net.kno5.kno.features
    features = net.kno5.kno.features
    phi52 = phi52.view(features).detach().numpy()
    counts = int(np.log2(features))
    q4 = QuantumRegister(counts,name='q4')
    qc4 = QuantumCircuit(q4)
    qc4.initialize(np.exp(1.j*np.asarray(phi52,dtype=np.complex128))/(np.power(2,counts/2,dtype=np.float64)), q4)
    for i in range(counts):
        qc4.rz(net.kno5.kno.weight.detach().numpy()[i] * p , q4[counts-1-i])
    job = backend.run(transpile(qc4,backend))
    state = job.result().get_statevector(qc4) # get_statevector 对应于 statevector_simulator
    phi52 = torch.from_numpy(np.array(state).astype(np.complex64)).angle()
    phi52 = phi52.view(b,c,h,w)
    x52 = r52 * torch.cos(phi52)
    
    # x11,x23,x32,x42,x52
    return net.dec(x11,x23,x32,x42,x52).squeeze().detach(),[qc0,qc1,qc2,qc3,qc4,]
    
    
def qkm_sim(input,p,net):
    '''
    Args:
        input: torch.tensor (3,h,h)
        p: int in [0,t-1]
    Returns:
        np.array -> net(input,p) (h,h)
        empty list
    '''
    net.eval()
    return net(input,p=torch.tensor(p).unsqueeze(-1)).squeeze().detach(),[]
           

def getv1v2v3(hint,vort,net,flag_qkm=0):
    '''
    Args:
        hint (2,h,h)
        vort (t,h,h)
        net (B,3,h,h)->(B,1,h,h)
        flag_qkm -> 0: qkm_sim 1: qkm_true
    Returns:
        v1: np.array (t,h,h) GT
        v2: np.array (t,h,h) AE
        v3: np.array (t,h,h) QKM
    '''    
    hint = torch.from_numpy(hint.astype(np.float32))
    v1 = torch.from_numpy(vort.astype(np.float32))

    h = hint.shape[-1]

    with torch.no_grad():
        # AE
        v2 = np.zeros_like(v1)
        for i in range(v1.shape[0]):
            input2=torch.zeros(3,h,h).unsqueeze(0)
            input2[0,:2,:,:]=hint[:2,:,:]
            input2[0,2,:,:]=v1[i,:,:]
            if flag_qkm==0:
                v2[i,:,:],_=qkm_sim(input=input2,p=0,net=net)
            else:
                v2[i,:,:],_=qkm_true(input=input2,p=0,net=net)
        # v2[0,:,:]=v[0,:,:]

        # QKM
        v3 = np.zeros_like(v1)
        input3=torch.zeros(3,h,h).unsqueeze(0)
        input3[0,:2,:,:]=hint[:2,:,:]
        input3[0,2,:,:]=v1[0,:,:]
        for i in range(v1.shape[0]):
            if flag_qkm==0:
                v3[i,:,:],_=qkm_sim(input=input3,p=i,net=net)
            else:
                v3[i,:,:],_=qkm_true(input=input3,p=i,net=net)
        # v3[0,:,:]=v[0,:,:]

    v1=v1.numpy()
    return v1,v2,v3 # np.array

def getv1v3_ex(hint,vort,net,step=10,flag_qkm=0):
    '''
    Args:
        hint (2,h,h)
        vort (t,h,h)
        net (B,3,h,h)->(B,1,h,h)
        flag_qkm -> 0: qkm_sim 1: qkm_true
    Returns:
        v1: np.array (t,h,h) GT
        v3: np.array (t,h,h) QKM
        v3_ex: np.array (step,h,h) QKM_ex
    '''
    hint = torch.from_numpy(hint.astype(np.float32))
    v1 = torch.from_numpy(vort.astype(np.float32))

    h = hint.shape[-1]

    with torch.no_grad():
        # QKM
        v3 = np.zeros_like(v1)
        input3=torch.zeros(3,h,h).unsqueeze(0)
        input3[0,:2,:,:]=hint[:2,:,:]
        input3[0,2,:,:]=v1[0,:,:]
        for i in range(v1.shape[0]):
            if flag_qkm==0:
                v3[i,:,:],_=qkm_sim(input=input3,p=i,net=net)
            else:
                v3[i,:,:],_=qkm_true(input=input3,p=i,net=net)
        v3_ex = np.zeros([step,v3.shape[-2],v3.shape[-1]])
        for i in range(step):
            if flag_qkm==0:
                v3_ex[i,:,:],_=qkm_sim(input=input3,p=(i+v1.shape[0]),net=net)
            else:
                v3_ex[i,:,:],_=qkm_true(input=input3,p=(i+v1.shape[0]),net=net)

    v1=v1.numpy()
    return v1,v3,v3_ex # np.array

def get_subprocess(hint,vort,net):
    '''
    Args:
        hint (2,h,h)
        vort (t,h,h)
        net (B,3,h,h)->(B,1,h,h)
    Returns:
        x11_list (np.array)
        x23_list (np.array)
        x32_list (np.array)
        x42_list (np.array)
        x52_list (np.array)
    '''        
    hint = torch.from_numpy(hint.astype(np.float32))
    v1 = torch.from_numpy(vort.astype(np.float32))
    t = v1.shape[0]
    h = hint.shape[-1]

    with torch.no_grad():
        # QKM
        v3 = np.zeros_like(v1)
        input3=torch.zeros(3,h,h).unsqueeze(0)
        input3[0,:2,:,:]=hint[:2,:,:]
        input3[0,2,:,:]=v1[0,:,:]
        
        # 中间层的shape
        r11,r23,r32,r42,r52 = net.encr(input3)
        phi11,phi23,phi32,phi42,phi52 = net.encphi(input3)
        print('net: r11.shape:',r11.shape)
        print('net: r23.shape:',r23.shape)
        print('net: r32.shape:',r32.shape)
        print('net: r42.shape:',r42.shape)
        print('net: r52.shape:',r52.shape)

        x11_list = torch.empty(t,16,128,128)
        x23_list = torch.empty(t,32,64,64)
        x32_list = torch.empty(t,64,32,32)
        x42_list = torch.empty(t,128,16,16)
        x52_list = torch.empty(t,256,8,8)

        x11_list[0,:,:,:] = (r11 * torch.cos(phi11)).squeeze(0)
        x23_list[0,:,:,:] = (r23 * torch.cos(phi23)).squeeze(0)
        x32_list[0,:,:,:] = (r32 * torch.cos(phi32)).squeeze(0)
        x42_list[0,:,:,:] = (r42 * torch.cos(phi42)).squeeze(0)
        x52_list[0,:,:,:] = (r52 * torch.cos(phi52)).squeeze(0)
        
        for i in range(v1.shape[0]):
            phi11 = net.kno1(phi11,p=torch.tensor(i).unsqueeze(-1))
            phi23 = net.kno2(phi23,p=torch.tensor(i).unsqueeze(-1))
            phi32 = net.kno3(phi32,p=torch.tensor(i).unsqueeze(-1))
            phi42 = net.kno4(phi42,p=torch.tensor(i).unsqueeze(-1))
            phi52 = net.kno5(phi52,p=torch.tensor(i).unsqueeze(-1))  
            x11_list[i,:,:,:] = (r11 * torch.cos(phi11)).squeeze(0)
            x23_list[i,:,:,:] = (r23 * torch.cos(phi23)).squeeze(0)
            x32_list[i,:,:,:] = (r32 * torch.cos(phi32)).squeeze(0)
            x42_list[i,:,:,:] = (r42 * torch.cos(phi42)).squeeze(0)
            x52_list[i,:,:,:] = (r52 * torch.cos(phi52)).squeeze(0)
            
    return x11_list.numpy(),x23_list.numpy(),x32_list.numpy(),x42_list.numpy(),x52_list.numpy() # np.array    
    
    
    
def getenergy(v):
    ''' 计算瞬时能谱
    Args:
        v (np.array): (t,h,h)
    Returns:
        k (np.array): (k,)
        e (np.array): (t,k)
    '''
    # 波数
    Nx=v.shape[-2] 
    Ny=v.shape[-1]
    kx = np.fft.fftfreq(Nx,d=1/Nx) # 频率范围是从 -N/2 到 N/2
    ky = np.fft.fftfreq(Ny,d=1/Ny)
    
    # 生成波数网格
    kx, ky = np.meshgrid(kx, ky)
    # 计算波数的模平方
    ksq = kx**2 + ky**2
    ksq[0,0] = 1. # 避免除以0
    # 能谱
    k=np.arange(np.ceil(np.sqrt(ksq.max())))+1
    e=np.zeros((v.shape[0],k.shape[0])) # (t,k)
    
    for t in range(v.shape[0]):
        # 频域内涡量
        vhat = np.fft.fft2(v[t,:,:]) 
        
        # 计算能量(频谱的能量应该在频域计算)
        vsq = 0.5 * (np.abs(vhat)**2) / ksq
        
        # 计算能谱
        et=np.zeros_like(k)
        for i in range(len(et)):
            et[i]+=np.sum(vsq[np.logical_and(ksq >= (k[i]-0.5)**2, 
                                            ksq < (k[i]+0.5)**2)])
        e[t,:]=et
 
    return k,e

# def getvelocity_psi(v,case):
#     ''' 二维不可压流, 由涡量场v,流函数的泊松方程计算速度场u (不准,不推荐)
#     Args:
#         v (np.array): (t,h,h)
#         case (int)
#     Returns:
#         ux (np.array): (t,h,h)
#         uy (np.array): (t,h,h)
#     '''    
#     if case == 0:
#         dx = 0.007843137718737125
#         dy = 0.003913894295692444
#     elif case == 1:
#         dx = 0.09817477207132212
#         dy = 0.09817477207132212
        
#     # 假设周期性边界，使用傅里叶谱方法
#     nx, ny = v.shape[-2],v.shape[-1]
#     kx = 2 * np.pi * np.fft.fftfreq(nx, dx)
#     ky = 2 * np.pi * np.fft.fftfreq(ny, dy)
#     kx, ky = np.meshgrid(kx, ky)
#     ksq = kx**2 + ky**2
#     ksq[0,0] = 1. # 避免除以0

#     # 解泊松方程 ∇²ψ = -ω
#     psi_hat = -fft2(v) / ksq
#     psi = np.real(ifft2(psi_hat))

#     # 计算速度场
#     ux = np.gradient(psi, dy, axis=0)  # u = ∂ψ/∂y
#     uy = -np.gradient(psi, dx, axis=1) # v = -∂ψ/∂x
#     return ux, uy

def getvelocity_bs(v):
    ''' 二维不可压流, 由涡量场v,Biot-Sarvart公式计算速度场u (推荐)
    Args:
        v (np.array): (t,h,h)
        case (int)
    Returns:
        ux (np.array): (t,h,h)
        uy (np.array): (t,h,h)
    '''    
    # 波数
    Nx=v.shape[-2] 
    Ny=v.shape[-1]
    kx = np.fft.fftfreq(Nx,d=1/Nx) # 频率范围是从 -N/2 到 N/2
    ky = np.fft.fftfreq(Ny,d=1/Ny)
    
    # 生成波数网格
    kx, ky = np.meshgrid(kx, ky)
    # 计算波数的模平方
    ksq = kx**2 + ky**2
    ksq[0,0] = 1. # 避免除以0
    
    # 速度
    ux = np.zeros_like(v)
    uy = np.zeros_like(v)
    
    for t in range(v.shape[0]):
        # 频域内涡量
        vhat = np.fft.fft2(v[t,:,:])
         
        # 在傅里叶空间中由Biot-Sarvart公式计算速度
        u_hat = (1j * ky * vhat) / ksq
        v_hat = (-1j * kx * vhat) / ksq   

        # 处理k=0模式 (设置为0)
        u_hat[0, 0] = 0
        v_hat[0, 0] = 0

        # 逆变换得到物理空间速度 (128^2 网格)
        ux[t,:,:] = np.real(ifft2(u_hat))
        uy[t,:,:] = np.real(ifft2(v_hat))

    return ux, uy

def getenergy_from_velocity(ux,uy):
    ''' 二维不可压流, 由速度场u计算能谱
    Args:
        ux (np.array): (t,h,h)
        uy (np.array): (t,h,h)
    Returns:
        k (np.array): (k,)
        e (np.array): (t,k)
    '''        
    # 波数
    Nx=ux.shape[-2] 
    Ny=ux.shape[-1]
    kx = np.fft.fftfreq(Nx,d=1/Nx) # 频率范围是从 -N/2 到 N/2
    ky = np.fft.fftfreq(Ny,d=1/Ny)
    # 生成波数网格
    kx, ky = np.meshgrid(kx, ky)
    # 计算波数的模平方
    ksq = kx**2 + ky**2
    ksq[0,0] = 1. # 避免除以0
    # 能谱
    k=np.arange(np.ceil(np.sqrt(ksq.max())))+1
    e=np.zeros((ux.shape[0],k.shape[0])) # (t,k)
    
    for t in range(ux.shape[0]):
        
        # 频域内速度
        ux_fft = np.fft.fft2(ux[t,:,:])
        uy_fft = np.fft.fft2(uy[t,:,:])
        
        # 计算能量(频谱的能量应该在频域计算)
        vsq = 0.5 * ( np.abs(ux_fft)**2 + np.abs(uy_fft)**2 )
        
        # 计算能谱
        et=np.zeros_like(k)
        for i in range(len(et)):
            et[i]+=np.sum(vsq[np.logical_and(ksq >= (k[i]-0.5)**2, 
                                            ksq < (k[i]+0.5)**2)])
        e[t,:]=et
    return k,e


def getstructure_vel(ux,uy,p_list=[2,3,4]):
    ''' 二维HIT (周期边界条件), 由速度场u计算瞬时结构函数
    Args:
        ux (np.array): (t,h,h)
        uy (np.array): (t,h,h)
        p_list (list[int]): 结构函数阶数
    Returns:
        dict: 包含结构函数计算结果，格式为：
            {
                'r': (np.array): (h-1,)
                's2': (np.array): (t,h-1)
                's3': (np.array): (t,h-1)
                ...
                }
            }
    '''       
    t,h,_ = ux.shape   # h = 128
    r = np.arange(1,h) # in [1,127]
    dictS = {f's{p:d}': np.zeros((t,h-1)) for p in p_list}
    dictS['r'] = r
    for i in range(len(r)):
        # 纵向结构函数 ux -> x ; uy -> y
        delta_ux = np.roll(ux,r[i],axis=-2) - ux
        delta_uy = np.roll(uy,r[i],axis=-1) - uy
        for p in p_list:
            dictS[f's{p:d}'][:,i] = 0.5*((delta_ux**p).mean(axis=(-2,-1)) 
                                 + (delta_uy**p).mean(axis=(-2,-1)))
    return dictS

def getstructure_vort(v,q_list=[0.5,1.0,1.5,2.0]):
    ''' 二维HIT (周期边界条件), 由涡量场v计算涡量的瞬时结构函数
    Args:
        v (np.array): (t,h,h)
        q_list (list[float]): 结构函数阶数为 2q
    Returns:
        dict: 包含结构函数计算结果，格式为：
            {
                'r': (np.array): (h-1,)
                's0.1': (np.array): (t,h-1) q=0.1 阶数为0.2
                's0.5': (np.array): (t,h-1) q=0.5 阶数为1.0
                ...
                }
            }
    '''       
    t,h,_ = v.shape   # h = 128
    r = np.arange(1,h) # in [1,127]
    dictS = {f's{q:.1f}': np.zeros((t,h-1)) for q in q_list}
    dictS['r'] = r
    for i in range(len(r)):
        # 结构函数
        delta_ux = np.abs(np.roll(v,r[i],axis=-2) - v)
        delta_uy = np.abs(np.roll(v,r[i],axis=-1) - v)
        for q in q_list:
            dictS[f's{q:.1f}'][:,i] = 0.5*((delta_ux**(q*2)).mean(axis=(-2,-1)) 
                                 + (delta_uy**(q*2)).mean(axis=(-2,-1)))
    return dictS
"""Reproduce the two phase/certificate illustrations in Section 4.

From the repository root, run: python phase_analysis.py
Plots are saved as PDF and PNG, and numerical summaries are saved under data/.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
DATA = ROOT / 'data'
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.grid':True,'grid.alpha':0.18,
                     'pdf.fonttype':42,'ps.fonttype':42,
                     'savefig.bbox':'tight'})
COLORS = ['#1675B6','#153C57','#00A779','#E77C29']


def gain(a, b, theta=0., alpha=1., beta=1., h=1.):
    a,b = np.broadcast_arrays(np.asarray(a),np.asarray(b))
    out=np.zeros(a.shape+(2,2))
    out[...,0,0]=a; out[...,0,1]=alpha*a*h
    out[...,1,0]=beta*b*((1+theta)*a+abs(theta))
    out[...,1,1]=b*(1+(1+theta)*alpha*beta*a*h)
    return out


def phase_figure():
    phase=np.linspace(0,np.pi,361)
    thetas=(-1.,0.,.5,1.)
    a=b=1/3
    fig,axs=plt.subplots(1,2,figsize=(7.15,2.75),layout='constrained')
    rows=[]
    for color,theta in zip(COLORS,thetas):
        L=gain(a,b,theta)
        q=float(np.max(abs(np.linalg.eigvals(L))))
        T=np.zeros((len(phase),2,2),complex)
        T[:,0,0]=a; T[:,0,1]=-a*np.exp(1j*phase)
        T[:,1,0]=b*((1+theta)*a-theta)
        T[:,1,1]=b*(1-(1+theta)*a*np.exp(1j*phase))
        rate=np.max(abs(np.linalg.eigvals(T)),axis=1)
        label=rf'$\theta={theta:g}$'
        axs[0].plot(phase/np.pi,rate,color=color,lw=1.8,label=label)
        axs[0].axhline(q,color=color,lw=1,ls='--',alpha=.65)
        axs[1].plot(phase/np.pi,q/rate,color=color,lw=1.8,label=label)
        rows.append({'theta':theta,'certified_factor':q,
                     'phase_zero_factor':float(rate[0]),
                     'phase_pi_factor':float(rate[-1]),
                     'minimum_ratio':float(np.min(q/rate)),
                     'maximum_ratio':float(np.max(q/rate))})
    axs[0].set(title='Exact factors and gain bounds',ylabel='Linear factor',xlabel=r'Coupling phase $\varphi/\pi$')
    axs[1].set(title='Certificate-to-exact ratio',ylabel=r'Certificate-to-exact ratio $\rho(L_\theta)/r_\theta(\varphi)$',xlabel=r'Coupling phase $\varphi/\pi$')
    for ax in axs:
        ax.set_xlim(0,1); ax.set_xticks(np.linspace(0,1,5))
    axs[0].legend(loc='upper left',ncol=2,frameon=False,fontsize=8)
    axs[1].axhline(1,color='#555555',lw=.7,ls=':')
    fig.savefig(FIG/'phase_resolved_gain_sharpness.pdf')
    fig.savefig(FIG/'phase_resolved_gain_sharpness.png', dpi=180)
    plt.close(fig)
    return rows


def soft(x,tau):
    return np.sign(x)*np.maximum(abs(x)-tau,0.)


def coupling(x,kind):
    if kind=='symmetric':
        return (2*x-np.roll(x,1,axis=-1)-np.roll(x,-1,axis=-1))/4
    return np.roll(x,1,axis=-1)


def one_calibration(kind,mu_a,mu_b,targets,initials):
    ngrid=len(mu_a); nstarts=targets[0].shape[0]
    a=np.repeat(mu_a,nstarts)[:,None]
    b=np.repeat(mu_b,nstarts)[:,None]
    ys=np.tile(targets[0],(ngrid,1)); xs=np.tile(targets[1],(ngrid,1))
    y=ys+np.tile(initials[0],(ngrid,1)); x=xs+np.tile(initials[1],(ngrid,1))
    ca=a*ys+coupling(xs,kind)+.12*np.sign(ys)
    db=b*xs-ys+.10*np.sign(xs)
    denom=np.maximum(1,np.sqrt(np.sum(ys*ys+xs*xs,axis=1)))
    err0=np.sqrt(np.sum((y-ys)**2+(x-xs)**2,axis=1))/denom
    cert_grid=mu_a*mu_b>1
    cert=np.repeat(cert_grid,nstarts)
    cert_ids=np.flatnonzero(cert)
    L=gain(1/(1+mu_a[cert_grid]),1/(1+mu_b[cert_grid]))
    eye=np.eye(2)[None,:,:]
    rhs=np.ones((len(L),2,1))
    r=np.linalg.solve(eye-L,rhs)[:,:,0]
    s=np.linalg.solve(eye-L.swapaxes(1,2),rhs)[:,:,0]
    p=s/r
    eta_r=np.max(np.einsum('nij,nj->ni',L,r)/r,axis=1)
    eta_s=np.max(np.einsum('nji,nj->ni',L,s)/s,axis=1)
    q_grid=np.sqrt(eta_r*eta_s)
    pp=np.repeat(p,nstarts,axis=0)
    history=np.full((len(cert_ids),100),np.nan)
    cert_lookup=np.full(len(x),-1,dtype=int);cert_lookup[cert_ids]=np.arange(len(cert_ids))
    # state: 0 unresolved, 1 reaches exact target, -1 numerical divergence
    state=np.zeros(len(x),dtype=int);active=np.arange(len(x))
    end=np.zeros(len(x),dtype=int)
    for k in range(1,5001):
        yy=soft(y[active]-coupling(x[active],kind)+ca[active],.12)/(1+a[active])
        xx=soft(x[active]+yy+db[active],.10)/(1+b[active])
        y[active]=yy;x[active]=xx
        dy=yy-ys[active];dx=xx-xs[active]
        sy=np.sum(dy*dy,axis=1);sx=np.sum(dx*dx,axis=1)
        err=np.sqrt(sy+sx)/denom[active]
        loc=cert_lookup[active];keep=loc>=0
        history[loc[keep],k%100]=np.log(np.sqrt(pp[loc[keep],0]*sy[keep]+pp[loc[keep],1]*sx[keep]))
        good=err<=1e-8
        bad=(~np.isfinite(err))|(err>1e6*np.maximum(1,err0[active]))
        state[active[good]]=1;state[active[bad]]=-1
        end[active[good|bad]]=k
        active=active[~(good|bad)]
        if not len(active):break
    end[active]=5000
    obs=np.full(len(cert_ids),np.nan)
    for j,idx in enumerate(cert_ids):
        if state[idx]!=1:continue
        stop=end[idx]
        start=max(int(.3*stop),stop-60,1)
        if stop-start<8:continue
        kk=np.arange(start,stop+1)
        vals=history[j,kk%100]
        if np.all(np.isfinite(vals)):
            obs[j]=np.exp(np.polyfit(kk,vals,1)[0])
    obs_grid=np.nanmedian(obs.reshape(-1,nstarts),axis=1)
    qrat=obs_grid/q_grid
    return {'kind':kind,'state':state.reshape(ngrid,nstarts),
            'fraction':(state==1).reshape(ngrid,nstarts).mean(axis=1),
            'q_cert':q_grid,'q_obs':obs_grid,
            'ratio_median':float(np.nanmedian(qrat)),
            'certified_runs':int(np.count_nonzero(cert)),
            'certified_converged':int(np.sum((state==1)&cert)),
            'total_runs':int(len(state)),
            'converged':int(np.sum(state==1)),
            'divergent':int(np.sum(state==-1)),
            'unresolved':int(np.sum(state==0))}


def calibration_figure():
    n=40;nstarts=8;seed=20261112
    rng=np.random.default_rng(seed)
    ya=rng.choice([-1.,1.],(nstarts,n))*rng.uniform(.5,1.,(nstarts,n))
    xa=rng.choice([-1.,1.],(nstarts,n))*rng.uniform(.5,1.,(nstarts,n))
    yi=2*rng.normal(size=(nstarts,n));xi=2*rng.normal(size=(nstarts,n))
    mus=np.geomspace(.15,4.,29)
    aa,bb=np.meshgrid(mus,mus)
    results=[]
    for kind in ['symmetric','nonsymmetric']:
        result=one_calibration(kind,aa.ravel(),bb.ravel(),(ya,xa),(yi,xi))
        results.append(result)
        print({k:v for k,v in result.items() if not isinstance(v,np.ndarray)},flush=True)
    fig=plt.figure(figsize=(7.15,4.55),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.05,1])
    top=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1])]
    bot=fig.add_subplot(gs[1,:])
    for ax,res,title in zip(top,results,['Symmetric cycle coupling','Nonsymmetric shift coupling']):
        im=ax.pcolormesh(mus,mus,res['fraction'].reshape(29,29),shading='nearest',vmin=0,vmax=1,cmap='Blues',rasterized=False,edgecolors='face',linewidth=.15)
        xx=np.geomspace(.25,4.,300)
        ax.plot(xx,1/xx,'--',color='#E77C29',lw=1.4)
        unresolved=(res['state']==0).any(axis=1).reshape(29,29)
        if unresolved.any():ax.scatter(aa[unresolved],bb[unresolved],s=7,c='#C55733',marker='x',lw=.6)
        ax.set(xscale='log',yscale='log',xlim=(.15,4),ylim=(.15,4),title=title,xlabel=r'$\mu_A$',ylabel=r'$\mu_B$')
        ax.grid(False)
    cb=fig.colorbar(im,ax=top,shrink=.85,pad=.01)
    cb.set_label('Fraction reaching target',fontsize=8)
    for col,res,label in zip([COLORS[0],COLORS[2]],results,['Symmetric coupling','Nonsymmetric coupling']):
        bot.scatter(res['q_cert'],res['q_obs'],s=12,c=col,alpha=.7,linewidths=0,label=label)
    bot.plot([0,1],[0,1],color='#777777',ls='--',lw=.9,label='Equal factors')
    bot.set(xlim=(0,1.015),ylim=(0,1.015),xlabel=r'Certified Hilbert-norm factor $q_P$',ylabel=r'Observed tail factor $q_{\mathrm{obs},P}$',title='Rate calibration in the same diagonal Hilbert norm')
    bot.legend(loc='upper left',frameon=False,fontsize=8,ncol=3)
    fig.savefig(FIG/'certificate_phase_rate_calibration.pdf')
    fig.savefig(FIG/'certificate_phase_rate_calibration.png', dpi=180)
    plt.close(fig)
    raw={}
    for res in results:
        for key,value in res.items():
            if isinstance(value,np.ndarray):raw[f"{res['kind']}_{key}"]=value
    raw.update(mu_a=aa,mu_b=bb)
    np.savez_compressed(DATA/'phase_calibration_data.npz',**raw)
    return {'dimension':n,'initializations':nstarts,'seed':seed,'grid_points_per_axis':29,
            'mu_range':[.15,4.],'maximum_iterations':5000,'target_tolerance':1e-8,
            'results':[{k:v for k,v in r.items() if not isinstance(v,np.ndarray)} for r in results]}


if __name__=='__main__':
    summary={'phase':phase_figure(),'calibration':calibration_figure()}
    (DATA/'phase_calibration_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

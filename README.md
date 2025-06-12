This code calculates the time-periodic linear response of a homogeneous and rapidly rotating Newtonian fluid enclosed within a sphere or spherical shell container. 

Consider a sphere/spherical shell with outer radius $R$ and inner radius $r$ (for a sphere $r = 0$).  The volume of the sphere is filled with an incompressible fluid of uniform density and temperature rotating with angular velocity $\Omega_0$ about a fixed axis $\hat{e}_z$.  The rotation axis defines the pole of the spherical-polar coordinate system $(r,\theta,\varphi)$ where $r$ is the radius, and $\theta$ and $\varphi$ are colatitude and longitude, respectively.  

The equations of motion in the rotating reference frame are given by,

$$\frac{d\vec{u}}{dt} + \vec{u} \cdot \nabla \vec{u} + 2 \Omega_0 \hat{e}_z \times \vec{u} = -\frac{1}{\rho}\nabla P + \nu \nabla^2 \vec{u}$$
$$ \nabla \cdot \vec{u} = 0$$

Where $\nu$ is the kinematic viscosity and $P$ is the reduced pressure, which includes the centrifugal force, the depth of the spherical shell is defined by $d = R-r$, which is used as the non-dimensional length scale within the numerical calculations.  It is natural to define the Ekman number, which is a non-dimensional parameter controlling the strength of the Coriolis force relative to viscosity. It is given by

$$E = \frac{\nu}{\Omega_0 d^2}$$

Using the rotational timescale $\Omega_0^{-1}$, the momentum equation can be written in terms of non-dimensional variables as,


$$\frac{d\vec{u}}{dt}+ \vec{u} \cdot \nabla \vec{u} + 2 \hat{e}_z \times \vec{u} = -\nabla P + E \nabla^2 \vec{u}$$

We neglect the non-linear term as quadratic in the small velocity field for weak forcing.  Furthermore, we restrict our attention to solutions with harmonic periodicity in time.  So we consider a fixed forcing/response frequency $\lambda$ and parametrize the velocity field as follows,

$$\vec{u} = \frac{1}{2} \vec{q}e^{i \lambda t}+\text{c.c.}$$

where $\vec{q}$ is a complex-valued field that depends only on spatial variables.  

To compute the numerical solutions, we use the following decomposition of the velocity field $\vec{q}$:

$$\vec{q} = \nabla \times (\nabla \times (W\hat{e}_z)) + \nabla \times (Z\hat{e}_z)$$

Where $W$ and $Z$ are the scalar poloidal and toroidal potential fields, respectively; this expansion automatically ensures that $\vec{q}$ satisfies mass conservation.  The scalar potentials are each expanded as a summation of fully normalized surface spherical harmonics $Y_\ell^m$ of degree $\ell$ and order $m$. 

$$ W = \sum_{\ell=1}^{\ell_{\text{max.}}} \sum_{m=-\ell}^\ell W_{\ell}^m(r) Y_{\ell}^m(\theta,\varphi)$$
$$ Z= \sum_{\ell=1}^{\ell_{\text{max.}}} \sum_{m=-\ell}^\ell Z_{\ell}^m(r) Y_{\ell}^m(\theta,\varphi)$$

Where $W_\ell^m(r)$ and $Z_\ell^m(r)$ are numerical coefficient functions and where

$$\int_0^{2\pi} \int_0^\pi Y_\ell^m(\theta,\varphi) Y_{\ell'}^{m'}(\theta,\varphi)^\dagger \sin\theta d\theta d\varphi = \delta_{\ell\ell'}\delta_{mm'}$$

By taking the operations $\hat{e}_r \cdot \nabla \times$ and $\hat{e}_r \cdot \nabla \times (\nabla \times)$ on the non-dimensionalized momentum equation, then making use of the orthogonality of the spherical harmonics we obtain coupled ODEs for each distinct pair $(\ell,m)$.

$$\tag{A.1}{A_\ell^m Z_\ell^m = C_\ell^m W_{\ell-1}^m + D_\ell^m W_{\ell+1}^m}$$
$$\tag{A.2}{A_\ell^m B_\ell^m W_\ell^m = C_\ell^m Z_{\ell-1}^m + D_\ell^m Z_{\ell+1}^m}$$

Where $A_\ell^m$, $B_\ell^m$, $C_\ell^m$, $D_\ell^m$ are operators defined by

$$A_\ell^m = i(\ell(\ell+1)\lambda - 2m) + \ell(\ell+1) E B_\ell^m$$
$$B_\ell^m = \frac{\ell(\ell+1)}{r^2} - \frac{d^2}{dr^2}$$
$$C_\ell^m = 2(\ell-1)(\ell+1) \sqrt{\frac{(\ell-m)(\ell+m)}{(2\ell-1)(2\ell+1)}} \left(\frac{d}{dr} - \frac{\ell}{r}\right)$$
$$D_\ell^m = 2\ell(\ell+2) \sqrt{\frac{(\ell+1-m)(\ell+1+m)}{(2\ell+1)(2\ell+3)}}\left(\frac{d}{dr} + \frac{\ell+1}{r}\right)$$

Because of the form of the coupling between coefficient functions in the equations of motion, the system decouples across spherical harmonic orders, meaning that each azimuthal symmetry order can be considered independently.  The coupling across spherical harmonic degree occurs in a nearest neighbor fashion, for example $Z_\ell^m$ couples only to $W_{\ell-1}^m$ and $W_{\ell+1}^m$, which suggests a further separation of the system into the sets of coefficients with odd degree toroidal coefficients and even degree poloidal coefficients and vis versa.   These sets of coefficients correspond to solutions that exhibit either odd or even symmetry about the equator.  For more details see [this page detailing the symmetry classes](https://github.com/imacph/spherical_flows/wiki/Symmetry-classes-of-the-coefficient-functions).

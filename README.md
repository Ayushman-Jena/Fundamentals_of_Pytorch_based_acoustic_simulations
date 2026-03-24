We start with neural network based acoustic wave simulation using Pytorch.
The 1 D wave Equation shall be our first attempt. Lets carry it forward from the very basics.

$$
\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2}
$$

Where, u(x,t)=displacement or pressure wave and c is the speed of wave. The steps shall include
1. use a neural network u(x,t)
2. Compute derivatives using Autograd
3. Minimize PDE residual + boundary + initial conditions 

Note: 1D Wave Equation
The 1D Wave Equation represents a balance between a wave's acceleration in time and its curvature in space. In simpler terms, it tells us that the more a medium (like a guitar string) is stretched or bent at a specific point, the faster it will snap back toward its equilibrium position.
Where:
*   $\frac{\partial^2 u}{\partial t^2}$: The **second partial derivative** of the displacement $u$ with respect to time $t$ (acceleration of the wave). The vertical acceleration of a point on the wave.
*   $c^2$: The **propagation speed** of the wave squared (a constant).
*   $\frac{\partial^2 u}{\partial x^2}$: The **second partial derivative** of the displacement $u$ with respect to position $x$ (the curvature of the wave). The "concavity" or curvature of the wave shape.


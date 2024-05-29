import numpy as np
import random
from DATT.python_utils import plotu, polyu
from DATT.refs.base_ref import BaseRef

class Lissajous2DRef(BaseRef):
    def __init__(self, altitude, should_yaw=False, t_end=10.0, seed=2023, env_diff_seed=False,  fixed_seed = False, **kwargs):
        offset_pos = kwargs.get('offset_pos', np.zeros(3))
        super().__init__(offset_pos)
 
        self.altitude = altitude
        self.t_end = t_end
        self.seed = seed
        self.env_diff_seed = env_diff_seed
        self.reset_count = 0
        self.fixed_seed = fixed_seed
        self.should_yaw = should_yaw
        self.coeff = None
        self.yaw_coeff = None

        np.random.seed(seed)
        self.reset()

    # Parameter Randomization for the Lissajous 2D Curve
    # A,B are the amplitudes of the x and y axis respectively
    # a,b are the frequencies of the x and y axis respectively
    # delta is a phase offset between the x and y axis
    def generate_coeff(self):
        A = np.random.uniform(-2, 2)
        B = np.random.uniform(-2, 2)
        frequency_choices = [0.25, 0.5, 0.75, 1, 1.5, 2]
        a = np.random.choice(frequency_choices)
        b = np.random.choice(frequency_choices)
        delta = np.random.uniform(-np.pi, np.pi)
        return [A, B, a, b, delta]

    def generate_yaw():
        C = np.random.uniform(-np.pi, np.pi)
        c = np.random.choice([0.25*np.pi, 0.5*np.pi, 0.75*np.pi])
        return [C, c]
    
    def reset(self):
        if self.fixed_seed:
            np.random.seed(self.seed)
        elif self.env_diff_seed and self.reset_count > 0:
            np.random.seed(random.randint(0, 1000000))

        self.coeff = self.generate_coeff()

        if self.should_yaw:
            self.yaw_coeff = self.generate_yaw()

        self.reset_count += 1

    def pos(self, t):
        A, B, a, b, delta = self.coeff
        x = A * np.sin(a * t + delta)
        y = B * np.sin(b * t)
        return np.array([
            x,
            y,
            t*0 + self.altitude
    ])

    def vel(self, t):
        A, B, a, b, delta = self.coeff
        x = a * A * np.cos(a * t + delta)
        y = b * B * np.cos(b * t)
        return np.array([
            x,
            y,
            t*0
            ])

    def acc(self, t):
        A, B, a, b, delta = self.coeff
        x = -a**2 * A * np.sin(a * t + delta)
        y = -b**2 * B * np.sin(b * t)
        return np.array([
            x,
            y,
            t*0
        ])

    def jerk(self, t):
        A, B, a, b, delta = self.coeff
        x = -a**3 * A * np.cos(a * t + delta)
        y = -b**3 * B * np.cos(b * t)
        return np.array([
            x,
            y,
            t*0
        ])

    def snap(self, t):
        A, B, a, b, delta = self.coeff
        x = a**4 * A * np.sin(a * t + delta)
        y = b**4 * B * np.sin(b * t)
        return np.array([
            x,
            y,
            t*0
        ])

    def yaw(self, t):
        if self.should_yaw:
            C, c = self.yaw_coeff
            return C * np.sin(c * t)
        return t*0

    def yawvel(self, t):
        if self.should_yaw:
            C, c = self.yaw_coeff
            return c * C * np.cos(c * t)
        return t*0

    def yawacc(self, t):
        if self.should_yaw:
            C, c = self.yaw_coeff
            return -c**2 * C * np.sin(c * t)
        return t*0
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ref = Lissajous2DRef(altitude=1.0, should_yaw=False, t_end=10.0)
    t = np.linspace(0, 10, 500)

    plt.subplot(2, 1, 1)
    plt.plot(t, ref.pos(t)[0, :], label='x')
    plt.subplot(2, 1, 2)
    plt.plot(t, ref.pos(t)[1, :], label='y')
    plt.savefig('lissajous2d.png')
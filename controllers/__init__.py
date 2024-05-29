from enum import Enum
from DATT.controllers.pid_controller import PIDController
from DATT.controllers.mppi_controller import MPPIController
from DATT.controllers.datt_controller import DATTController
from DATT.controllers.gc_controller import GCController
from DATT.controllers.cntrl_config import *
class ControllersZoo(Enum):
    PID = 'pid'
    MPPI = 'mppi'
    DATT = 'datt'
    GC = 'gc'
    
    def cntrl(self, config, cntrl_configs : dict):
        pid_config = cntrl_configs.get('pid', PIDConfig())
        mppi_config = cntrl_configs.get('mppi', MPPIConfig())
        datt_config = cntrl_configs.get('datt', DATTConfig())
        gc_config = cntrl_configs.get('gc', GCConfig())
        

        return {
            ControllersZoo.PID : PIDController(config, pid_config),
            ControllersZoo.MPPI : MPPIController(config, mppi_config),
            ControllersZoo.DATT : DATTController(config, datt_config),
            ControllersZoo.GC : GCController(config, gc_config)

        }[ControllersZoo(self._value_)]
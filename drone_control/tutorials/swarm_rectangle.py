import time
import logging

import cflib.crtp
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.swarm import Swarm

uris = {
    'radio://0/80/250K/E7E7E7E7E3',
    'radio://0/80/250K/E7E7E7E7E4',
    # Add more URIs if you want more copters in the swarm
    # URIs in a swarm using the same radio must also be on the same channel
}

logging.basicConfig(level=logging.CRITICAL)


def activate_led_bit_mask(scf):
    scf.cf.param.set_value('led.bitmask', 255)

def deactivate_led_bit_mask(scf):
    scf.cf.param.set_value('led.bitmask', 0)

def light_check(scf):
    activate_led_bit_mask(scf)
    time.sleep(2)
    deactivate_led_bit_mask(scf)




def take_off(scf):
    commander= scf.cf.high_level_commander

    commander.takeoff(1.0, 2.0)
    time.sleep(3)

def land(scf):
    commander= scf.cf.high_level_commander

    commander.land(0.0, 2.0)
    time.sleep(2)

    commander.stop()

def hover_sequence(scf):
    take_off(scf)
    land(scf)




def run_square_sequence(scf):
    box_size = 1
    flight_time = 2

    commander= scf.cf.high_level_commander

    commander.go_to(box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(-box_size, 0, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)

    commander.go_to(0, -box_size, 0, 0, flight_time, relative=True)
    time.sleep(flight_time)




if __name__ == '__main__':
    cflib.crtp.init_drivers()
    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(light_check)
        try:
            swarm.reset_estimators()
        except Exception as e:
            print("Estimator reset failed:", e)

        # we want only one copter at a time executing the hover_sequence
        swarm.sequential(hover_sequence)

        # take off and land in sync, you can use the parallel_safe()
        # swarm.parallel_safe(take_off)
        # swarm.parallel_safe(run_square_sequence)
        # swarm.parallel_safe(land)
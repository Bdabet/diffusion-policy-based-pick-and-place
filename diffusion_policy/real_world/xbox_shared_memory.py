import multiprocessing as mp
import numpy as np
import time
# from spnav import spnav_open, spnav_poll_event, spnav_close, SpnavMotionEvent, SpnavButtonEvent
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from diffusion_policy.real_world.xbox_controller import XboxController

class Spacemouse(mp.Process):
    def __init__(self, 
            shm_manager, 
            get_max_k=30, 
            frequency=200,
            max_value=500, 
            deadzone=(0,0,0,0,0,0), 
            dtype=np.float32,
            n_buttons=2,
            ):
        """
        Continuously listen to 3D connection space naviagtor events
        and update the latest state.

        max_value: {300, 500} 300 for wired version and 500 for wireless
        deadzone: [0,1], number or tuple, axis with value lower than this value will stay at 0
        
        front
        z
        ^   _
        |  (O) space mouse
        |
        *----->x right
        y
        """
        super().__init__()
        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # copied variables
        self.frequency = frequency
        self.max_value = max_value
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons
        # self.motion_event = SpnavMotionEvent([0,0,0], [0,0,0], 0)
        # self.button_state = defaultdict(lambda: False)
        self.tx_zup_spnav = np.array([
            [0,0,-1],
            [1,0,0],
            [0,1,0]
        ], dtype=dtype)

        example = {
            # 3 translation, 3 rotation, 1 period
            'motion_event': np.zeros((7,), dtype=np.int64),
            # left and right button
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager, 
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        print("mod space mouse intialized")

        # shared variables
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= get state APIs ==========

    def get_motion_state(self):
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], 
            dtype=self.dtype) / self.max_value
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state
    
    def get_motion_state_transformed(self):
        """
        Return in right-handed coordinate
        z
        *------>y right
        |   _
        |  (O) space mouse
        v
        x
        back

        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_zup_spnav @ state[:3]
        tf_state[3:] = self.tx_zup_spnav @ state[3:]
        return tf_state

    def get_button_state(self):
        state = self.ring_buffer.get()
        return state['button_state']
    
    def is_button_pressed(self, button_id):
        return self.get_button_state()[button_id]
    
    #========== start stop API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()
    
    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= main loop ==========
    def run(self):
        

        self.motion_event = np.zeros((7,), dtype=np.int64)  # only using indices 0–2
        self.button_state = np.zeros((self.n_buttons,), dtype=bool) #dummy
        # send one message immediately so client can start reading
        self.ring_buffer.put({
            'motion_event': self.motion_event.copy(),
            'button_state': self.button_state.copy(),
            'receive_timestamp': time.time()
        })
        # print("in sm.run")
        self.ready_event.set()

        joystick = XboxController()

        

        
        while not self.stop_event.is_set():
            
            # print("joystick_read",joystick.read())
            

            self.motion_event[0] = joystick.read()[1]*100  # X
            self.motion_event[1] = joystick.read()[0]*100  # Y
            if joystick.read()[2] > 0:
                self.motion_event[2] = 50
            elif joystick.read()[3] > 0:
                self.motion_event[2] = -50
            else:
                self.motion_event[2] = 0

            # self.motion_event[2] = 0                   # Z unused
            self.motion_event[6] = 1                   # dummy

            # print(f"motion_event: {self.motion_event}")

            self.ring_buffer.put({
                'motion_event': self.motion_event.copy(),
                'button_state': self.button_state.copy(),
                'receive_timestamp': time.time()
            })
            

            time.sleep(1 / self.frequency)
    
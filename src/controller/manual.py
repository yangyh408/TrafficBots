import keyboard
import time

class ManualController():
    def __init__(self):
        self.acc_status = None
        self.lane_change_status = None
        self.speed_limit = 40
        self.acc_limit = 2
        self.rotate_limit = 0.002
        self.sleep_time = 0

    def _pressed_key(self):
        up_or_down = None
        left_or_right = None

        if keyboard.is_pressed('up') and not keyboard.is_pressed('down'):
            up_or_down = 'up'
        if keyboard.is_pressed('down') and not keyboard.is_pressed('up'):
            up_or_down = 'down'

        if keyboard.is_pressed('left') and not keyboard.is_pressed('right'):
            left_or_right = 'left'
        if keyboard.is_pressed('right') and not keyboard.is_pressed('left'):
            left_or_right = 'right'

        return (up_or_down, left_or_right)
    
    def act(self, agent_state, agent_size, agent_valid):
        acc = 0
        rotate = 0

        up_or_down, left_or_right = self._pressed_key()

        if up_or_down == 'up':
            acc = self.acc_limit
        if up_or_down == 'down' and agent_state[0, 0, 3] > 0.1:
            acc = -self.acc_limit

        if left_or_right == 'left':
            rotate = -self.rotate_limit
        elif left_or_right == 'right':
            rotate = self.rotate_limit

        time.sleep(self.sleep_time)
        
        return [acc, rotate]
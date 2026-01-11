import numpy as np

class ControllerInterface:
    """
    WOUSE V1.2 NREL 5MW Baseline Controller
    """
    def __init__(self, dt):
        self.dt = dt
        self.rated_rpm = 12.1
        self.rated_power = 5e6
        
    def update(self, t, rotor_speed_rpm, pitch_angle_deg, wind_speed):
        """
        [V1.2 Interface]
        t: Time (s)
        rotor_speed_rpm: Rotor Speed (rpm)
        pitch_angle_deg: Current platform pitch (or blade pitch if feedback available)
        wind_speed: Hub height wind speed (m/s)
        """
        
        # --- Simple Logic for V1.2 ---
        
        # 1. Torque Control (Rated)
        gen_torque = 43093.55 
        
        # 2. Pitch Control (Open Loop based on Wind Speed)
        # Standard Region 3 control logic placeholder
        pitch_cmd = 0.0
        
        rated_wind = 11.4
        if wind_speed > rated_wind:
            # Linear sensitivity approx 2.3 deg/m/s for NREL 5MW
            pitch_cmd = (wind_speed - rated_wind) * 2.3
            if pitch_cmd > 90: pitch_cmd = 90.0
        else:
            pitch_cmd = 0.0
            
        return gen_torque, pitch_cmd

import configparser

class CamParam():
    def __init__(self,camera_sn,set_resolution='HD') :
        self.sn=camera_sn
        self.resolution=set_resolution
        self.Baseline=None
        self.f_pixel=None
        self.CxCy=None
    def get_params(self):    
        # Create a ConfigParser object
        config = configparser.ConfigParser()
        # Read the configuration file
        config.read(f'PoseEstimation\config\SN{self.sn}.conf')

        # left cam parameters
        left_fx_value = float(config.get(f'LEFT_CAM_{self.resolution}', 'fx'))
        left_fy_value = float(config.get(f'LEFT_CAM_{self.resolution}', 'fy'))
        left_cx_value = float(config.get(f'LEFT_CAM_{self.resolution}', 'cx'))
        left_cy_value = float(config.get(f'LEFT_CAM_{self.resolution}', 'cy'))

        # right cam parameters
        right_fx_value = float(config.get(f'RIGHT_CAM_{self.resolution}', 'fx'))
        right_fy_value = float(config.get(f'RIGHT_CAM_{self.resolution}', 'fy'))
        right_cx_value = float(config.get(f'RIGHT_CAM_{self.resolution}', 'cx'))
        right_cy_value = float(config.get(f'RIGHT_CAM_{self.resolution}', 'cy'))
        
        self.Baseline = float(config.get(f'STEREO', 'Baseline'))
        self.CxCy={"Left_Cx":left_cx_value,"Left_Cy":left_cy_value,"Right_Cx":right_cx_value,"Right_Cy":right_cy_value}
        self.f_pixel=(left_fx_value+left_fy_value+right_fx_value+right_fy_value)/4
       



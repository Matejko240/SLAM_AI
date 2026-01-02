import sys
if sys.prefix == 'C:\\Users\\rjane\\AppData\\Local\\Programs\\Python\\Python310':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '\\\\wsl.localhost\\Ubuntu\\home\\matejko\\SLAM_AI\\install\\ai_slam_eval'

import os
import sys

# Get the absolute path of the root directory
global_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),  ".."))

# Add the root directory to the Python Path
sys.path.append(global_root_dir)
                                  
                                  
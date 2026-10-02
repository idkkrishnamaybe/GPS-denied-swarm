import os                     # <-- ADD THIS
from glob import glob         # <-- ADD THIS
from setuptools import find_packages, setup

package_name = 'robofest'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        
        # ADD THIS LINE TO INSTALL YOUR LAUNCH FILES
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vaibhav',
    maintainer_email='your_email@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_localizer = robofest.aruco_localizer:main',
            'swarm_mission = robofest.swarm_mission:main',
        ],
    },
)

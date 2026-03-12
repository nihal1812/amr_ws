from setuptools import setup, find_packages

package_name = 'ekf_slam'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='nihal',
    maintainer_email='nihal@todo.todo',
    description='Pure EKF-SLAM implementation',
    license='MIT',
)

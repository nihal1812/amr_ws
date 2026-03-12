from setuptools import find_packages, setup

package_name = 'amr_navigation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nihal',
    maintainer_email='nihal@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
                'global_planner_node = amr_navigation.global_planner_node:main',
                'pure_pursuit_controller = amr_navigation.pure_pursuit_controller:main',
                'dwa_controller = amr_navigation.dwa_controller:main',
                'amr_controller = amr_navigation.amr_controller:main',
        ],
    },
)

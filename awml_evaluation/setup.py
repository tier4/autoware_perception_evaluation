from setuptools import setup

package_name = 'awml_evaluation'

setup(
    name=package_name,
    version='1.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Satoshi Tanaka',
    maintainer_email='satoshi.tanaka@tier4.jp',
    description='Autoware Machine Learning Evaluator',
    license='Apache V2',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)

from setuptools import setup

package_name = "awml_evaluation"
pm = package_name
pe = pm + "/evaluation"

setup(
    name=package_name,
    version="1.3.1",
    packages=[
        pm,
        pm + "/common",
        pm + "/util",
        pm + "/visualization",
        pe,
        pe + "/matching",
        pe + "/metrics",
        pe + "/metrics/detection",
        pe + "/result",
        pe + "/sensing",
    ],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Satoshi Tanaka",
    maintainer_email="satoshi.tanaka@tier4.jp",
    description="Autoware Machine Learning Evaluator",
    license="Apache V2",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)

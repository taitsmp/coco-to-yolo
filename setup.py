from setuptools import setup, find_packages

setup(
    name="coco-to-yolo",
    version="0.1.3",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'coco_to_yolo=coco_to_yolo.coco_to_yolo:main',
        ],
    },
    install_requires=[
        "PyYAML>=5.1",
    ],
    author="Tait Larson",
    author_email="telarson@gmail.com",
    description="Tools for working with COCO and YOLO format datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/taitsmp/coco-to-yolo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
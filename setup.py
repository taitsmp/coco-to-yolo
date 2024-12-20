from setuptools import setup, find_packages

setup(
    name="coco-yolo-tools",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyYAML>=5.1",
    ],
    entry_points={
        'console_scripts': [
            'coco-to-yolo=coco-to-yolo.coco_to_yolo:main',
            'filter-coco=coco-to-yolo.filter_coco_annotations:main',
        ],
    },
    author="Tait Larson",
    author_email="telarson@gmail.com",
    description="Tools for working with COCO and YOLO format datasets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/taitsmp/coco-yolo-tools",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)

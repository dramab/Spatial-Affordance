from setuptools import setup, find_packages

setup(
    name="spatial_affordance",
    version="0.1.0",
    description="End-to-end 3D Visual Grounding: RGB + PointCloud + Text -> 3D BBox",
    author="",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "transformers>=4.35.0",
        "open3d>=0.17.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
    ],
)

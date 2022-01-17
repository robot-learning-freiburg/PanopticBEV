from os import path, listdir

import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def find_sources(root_dir):
    sources = []
    for file in listdir(root_dir):
        _, ext = path.splitext(file)
        if ext in [".cpp", ".cu"]:
            sources.append(path.join(root_dir, file))

    return sources


def make_extension(name, package):
    return CUDAExtension(
        name="{}.{}._backend".format(package, name),
        sources=find_sources(path.join("src", name)),
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": ["--expt-extended-lambda"],
        },
        include_dirs=["include/"],
    )

here = path.abspath(path.dirname(__file__))

setuptools.setup(
    # Meta-data
    name="PanopticBEV",
    author="Nikhil Gosala",
    author_email="gosalan@cs.uni-freiburg.de",
    description="PanopticBEV Model Code",
    version="1.0.0",
    url="http://panoptic-bev.cs.uni-freiburg.de/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],

    python_requires=">=3, <4",

    # Package description
    packages=[
        "panoptic_bev",
        "panoptic_bev.algos",
        "panoptic_bev.config",
        "panoptic_bev.data",
        "panoptic_bev.models",
        "panoptic_bev.modules",
        "panoptic_bev.modules.heads",
        "panoptic_bev.utils",
        "panoptic_bev.utils.bbx",
        "panoptic_bev.utils.nms",
        "panoptic_bev.utils.parallel",
        "panoptic_bev.utils.roi_sampling",
    ],
    ext_modules=[
        make_extension("nms", "panoptic_bev.utils"),
        make_extension("bbx", "panoptic_bev.utils"),
        make_extension("roi_sampling", "panoptic_bev.utils")
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)

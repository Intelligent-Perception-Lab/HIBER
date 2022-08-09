'''
 # @ Author: Zhi Wu
 # @ Create Time: 2022-08-09 12:36:35
 # @ Modified by: Zhi Wu
 # @ Modified time: 2022-08-09 12:37:06
 # @ Description: Install scripts.
 '''

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='HIBERTools',
    version='1.0.0',
    author='Zhi Wu',
    author_email='wzwyyx@mail.ustc.edu.cn',
    description='Tools of HIBER Dataset',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/wuzhiwyyx/HIBER/tree/package/HIBERTools',
    project_urls = {
        "Bug Tracker": "https://github.com/wuzhiwyyx/HIBER/issues"
    },
    license='MIT',
    packages=['HIBERTools'],
    install_requires=['numpy', 'lmdb', 'tqdm', 'opencv-python', 'matplotlib'],
)
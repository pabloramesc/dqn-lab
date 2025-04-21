from setuptools import setup, find_packages

setup(
    name='dqn',
    version='0.1.0',
    packages=find_packages(include=['dqn', 'dqn.*']),
    install_requires=[
        'ale_py',
        'gym',
        'gymnasium',
        'keras',
        'matplotlib',
        'numpy',
        'opencv_python',
        'setuptools',
        'tensorflow',
    ],
    description='A Deep Q-Network (DQN) module for reinforcement learning.',
    author='Pablo Ramirez',
    author_email='ramirez.escudero.pablo@gmail.com',
    url='https://github.com/pabloramesc/dqn-lab',  # Update with the actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
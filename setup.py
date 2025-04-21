from setuptools import setup, find_packages

setup(
    name='dqn',
    version='0.1.0',
    packages=find_packages(include=['dqn', 'dqn.*']),
    install_requires=[
        'ale_py==0.10.2',
        'gym==0.26.2',
        'gymnasium==1.1.1',
        'keras==3.9.0',
        'matplotlib==3.10.1',
        'numpy==2.2.5',
        'opencv_python==4.11.0.86',
        'setuptools==76.0.0',
        'tensorflow==2.16.1',
    ],
    description='A Deep Q-Network (DQN) module for reinforcement learning.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/dqn-lab',  # Update with the actual URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
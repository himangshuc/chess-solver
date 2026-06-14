from setuptools import setup, find_packages

setup(
    name="chess-board-component",
    version="0.1.0",
    packages=find_packages(),
    package_data={"chess_board": ["frontend/index.html"]},
    install_requires=["streamlit>=1.37"],
)

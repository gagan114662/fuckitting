"""
Setup script for AlgoForge 3.0 - Claude Code SDK powered quantitative trading system
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="algoforge-3",
    version="3.0.0",
    author="AlgoForge Team",
    author_email="contact@algoforge.ai",
    description="Intelligent quantitative trading system powered by Claude Code SDK and QuantConnect",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/algoforge/algoforge-3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "claude-code-sdk>=0.1.0",
        "requests>=2.31.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "yfinance>=0.2.18",
        "ta>=0.10.2",
        "scipy>=1.11.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
        "aiohttp>=3.8.0",
        "websockets>=11.0.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "schedule>=1.2.0",
        "python-dateutil>=2.8.0",
        "pytz>=2023.3"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "algoforge=algoforge_main:main",
        ],
    },
)
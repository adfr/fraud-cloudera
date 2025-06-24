# Cloudera AI Jobs Template

This template provides a comprehensive framework for setting up and managing jobs in Cloudera Machine Learning (CML), including ML workflows, data processing, and scheduled tasks. It now includes enhanced environment variable support and a generic application launcher.

## 🚀 Quick Start

1. Clone this repository into your CML project
2. Set the `TEMPLATE_DIR` environment variable if your project structure differs
3. Copy `.env.example` to `.env` and fill in your values (if using the job runner)
4. Run `python app_setup.py` to launch the demo Flask application
5. Or run `python run_jobs.py` to create all jobs in CML

## 📁 Project Structure

```
cloudera-AI-template/
├── config/
│   └── jobs_config.yaml      # Job configurations
├── scripts/
│   ├── hello_world.py        # Basic job example
│   ├── example_job.py        # Environment activation example  
│   └── app.py               # Demo Flask application
├── template/
│   ├── scripts/
│   │   └── hello_world.py   # Production hello world script
│   └── requirements.txt     # Template requirements
├── src/                      # Additional source code
├── results/                  # Job outputs (created automatically)
├── create_environment.py     # Environment setup script (uses uv)
├── app_setup.py             # Generic app launcher (supports uv/pip)
├── run_jobs.py              # Main job creation script
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```

## 🔧 New Features

### Environment Variable Support
- **TEMPLATE_DIR**: Configurable template directory name (default: "template")
- All scripts now dynamically use this variable instead of hardcoded paths
- Compatible with different GitHub repository names and structures

### Generic App Launcher (app_setup.py)
- Supports both UV and pip package managers
- Automatically creates virtual environments
- Installs requirements and launches applications
- Cross-platform compatibility (Unix/macOS focused)
- Configurable for any Python application

### Enhanced Environment Setup
- Uses UV package manager for faster dependency installation
- Falls back to pip if UV is unavailable
- Creates project_env virtual environment
- IPython/Jupyter compatible (no sys.exit() issues)

## ⚙️ Configuration

### Environment Variables

Set these in your shell or CI/CD environment:

```bash
# Template directory configuration
export TEMPLATE_DIR=template  # or your custom directory name

# CML API Configuration (for job runner)
export CML_API_HOST=https://ml-12345.cloud.example.com
export CML_API_KEY=your_api_key_here
export CML_PROJECT_ID=project_id_here

# ML Runtime ID (required for ML Runtime projects)
export CML_RUNTIME_ID=python3.11

# Default resource settings (optional)
export DEFAULT_CPU=1
export DEFAULT_MEMORY=2
export DEFAULT_TIMEOUT=3600

# Flask app configuration
export CDSW_READONLY_PORT=8090  # Port for web applications
```

### Job Configuration (jobs_config.yaml)

Each job has the following structure:

```yaml
jobs:
  job_key:
    name: Human-readable job name
    script: template/path/to/script.py  # Automatically adjusted based on TEMPLATE_DIR
    kernel: python3
    runtime_id: 91
    cpu: 4
    memory: 8
    timeout: 3600
    environment:
      ENV_VAR: value
    arguments: --param value
    parent_job_id: previous_job  # For dependent jobs
    schedule: "0 8 * * 1"       # For scheduled jobs (cron format)
```

## 📋 Included Jobs

1. **create_env** - Sets up the Python environment with required packages using UV
2. **scheduled_report** - Weekly scheduled job example

## 🎯 Application Examples

### Demo Flask App (scripts/app.py)
- Simple web application demonstrating best practices
- Environment variable integration
- Health check and API endpoints
- Ready-to-use template for web applications

### Usage:
```bash
# Launch with automatic setup
python app_setup.py

# Or run directly
python scripts/app.py

# Access at: http://127.0.0.1:8090
```

## 🔧 Key Features

- **Environment Variable Support** - Configurable paths and settings
- **UV Package Manager** - Fast dependency installation with pip fallback
- **IPython Compatible** - Works in both script and interactive environments  
- **Generic App Launcher** - Reusable setup script for any Python application
- **Automatic path detection** - Works with different directory structures
- **Dependency management** - Jobs can depend on other jobs
- **Flexible scheduling** - Support for cron-style scheduling
- **Resource configuration** - CPU, memory allocation
- **Environment variables** - Pass configuration to jobs
- **Cross-platform** - Unix/macOS focused (Windows paths removed)

## 📝 Adding New Jobs

1. Create your script in the appropriate directory:

```python
#!/usr/bin/env python3
"""Your job description"""

import os
import argparse
import logging

def main():
    # Get template directory from environment
    template_dir = os.environ.get("TEMPLATE_DIR", "template")
    
    # Your job logic here
    pass

if __name__ == "__main__":
    main()
```

2. Add the job configuration to `config/jobs_config.yaml`:

```yaml
jobs:
  my_new_job:
    name: My New Job
    script: template/scripts/my_new_job.py  # Will be adjusted automatically
    kernel: python3
    runtime_id: 91
    cpu: 2
    memory: 4
    parent_job_id: create_env
```

3. Run `python run_jobs.py` to create the job in CML

## 🏃 ML Runtimes vs. Engine Runtimes

CML supports two types of runtimes:

1. **ML Runtimes** (newer) - Require a `runtime_id` parameter
2. **Engine Runtimes** (legacy) - Don't require a `runtime_id`

For ML Runtime projects, specify the `runtime_id` either:
- In each job configuration, or
- As `CML_RUNTIME_ID` in the environment variables (default for all jobs)

## 🛠️ Package Managers

The template supports both UV and pip:

- **UV** (default): Fast Python package installer and resolver
- **pip** (fallback): Traditional Python package manager

Configure in `app_setup.py`:
```python
USE_UV = True  # Set to False to use pip instead
```

## 🔍 Debugging

1. Check job logs in the CML UI
2. Use `--log_level DEBUG` in job arguments for verbose logging
3. Verify environment variables are set correctly:
   ```bash
   echo $TEMPLATE_DIR
   echo $CML_API_HOST
   ```
4. Ensure all file paths are correct
5. Test the demo app with `python app_setup.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Update documentation as needed
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues or questions:
- Check the CML documentation
- Open an issue in this repository
- Contact your CML administrator

## 🔄 Migration Guide

If upgrading from an older version:

1. Set the `TEMPLATE_DIR` environment variable if your directory structure differs
2. Update any hardcoded "template" paths in custom scripts
3. Install UV for faster package management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
4. Test the new app launcher: `python app_setup.py`
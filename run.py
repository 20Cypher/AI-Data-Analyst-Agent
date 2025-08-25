#!/usr/bin/env python3
"""Startup script for AI Data Analyst Agent"""
import os
import sys
import subprocess
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_environment():
    """Check if required environment is set up"""
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning(".env file not found. Creating from .env.example...")
        example_file = Path('.env.example')
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            logger.info("Created .env file. Please edit it with your OpenAI API key.")
            return False
        else:
            logger.error("Neither .env nor .env.example found!")
            return False
    
    # Check if OpenAI API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY not found in .env file!")
        logger.info("Please add your OpenAI API key to the .env file:")
        logger.info("OPENAI_API_KEY=your_key_here")
        return False
    
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], 
                      check=True, capture_output=True)
        logger.info("Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def setup_directories():
    """Create required directories"""
    directories = ['data', 'artifacts', 'artifacts/charts']
    for dir_path in directories:
        Path(dir_path).mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_path}")

def copy_sample_data():
    """Copy sample data if data directory is empty"""
    data_dir = Path('data')
    if not any(data_dir.glob('*.csv')):
        logger.info("No datasets found. Sample datasets should be in data/ directory.")
        logger.info("You can upload datasets via the web interface.")

def run_server(host='0.0.0.0', port=8000, reload=False):
    """Run the FastAPI server"""
    logger.info(f"Starting server on {host}:{port}")
    
    try:
        import uvicorn
        uvicorn.run(
            "web.backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except ImportError:
        logger.error("uvicorn not installed. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'uvicorn[standard]'])
        import uvicorn
        uvicorn.run(
            "web.backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Data Analyst Agent")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies before running')
    parser.add_argument('--skip-checks', action='store_true', help='Skip environment checks')
    
    args = parser.parse_args()
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Check environment unless skipped
    if not args.skip_checks:
        if not check_environment():
            sys.exit(1)
    
    # Copy sample data
    copy_sample_data()
    
    # Print startup info
    logger.info("="*60)
    logger.info("AI Data Analyst Agent Starting Up")
    logger.info("="*60)
    logger.info(f"Web Interface: http://localhost:{args.port}")
    logger.info(f"API Documentation: http://localhost:{args.port}/docs")
    logger.info("="*60)
    
    # Run server
    run_server(args.host, args.port, args.reload)

if __name__ == "__main__":
    main()
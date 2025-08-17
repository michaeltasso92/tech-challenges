# Docker Testing Guide

This guide covers how to test Docker builds and runtime execution for all AI microservices.

## Overview

The AI microservices are containerized using Docker and can be tested using the `test_docker.py` script. This script:

- ✅ Tests Docker builds for each service
- ✅ Optionally tests runtime execution
- ✅ Provides detailed feedback and error reporting
- ✅ Automatically cleans up test images
- ✅ Supports individual service testing

## Services with Docker Support

The following services have Dockerfiles and can be tested:

- **api**: FastAPI recommendation service
- **ui**: Streamlit web interface
- **parser**: JSON guideline parsing service
- **trainer**: PMI-based model training
- **trainer-bienc**: Bi-encoder model training
- **featurizer**: Text featurization service

## Quick Start

### Prerequisites

1. **Docker installed and running**
   ```bash
   docker --version
   ```

2. **Python 3.8+** (for running the test script)

### Basic Usage

#### Test All Services (Build Only)
```bash
cd AI
python test_docker.py
```

#### Test All Services (Build + Runtime)
```bash
python test_docker.py --runtime
```

#### Test Specific Service
```bash
python test_docker.py --service api
python test_docker.py --service ui --runtime
```

#### Verbose Output
```bash
python test_docker.py --verbose
```

## Command Line Options

| Option | Description |
|--------|-------------|
| `--service SERVICE` | Test specific service only |
| `--verbose, -v` | Show detailed build output |
| `--runtime` | Test runtime execution (slower) |
| `--no-cleanup` | Don't clean up test images |
| `--timeout SECONDS` | Build timeout (default: 300) |

## Test Types

### 1. Build Tests
- ✅ Verifies Dockerfile syntax
- ✅ Tests dependency installation
- ✅ Validates file copying and permissions
- ✅ Fast execution (~30-60 seconds per service)

### 2. Runtime Tests
- ✅ Tests container startup
- ✅ Validates command execution
- ✅ Tests basic functionality
- ✅ Slower execution (~1-2 minutes per service)

## Example Output

```
🐳 Testing Docker builds for AI microservices...

🔍 Checking Docker availability...
✅ Docker available: Docker version 20.10.21

📋 Testing services: api, featurizer, parser, trainer, trainer-bienc, ui

🔨 Building api...
✅ api built successfully

🔨 Building featurizer...
✅ featurizer built successfully

🔨 Building parser...
✅ parser built successfully

🔨 Building trainer...
✅ trainer built successfully

🔨 Building trainer-bienc...
✅ trainer-bienc built successfully

🔨 Building ui...
✅ ui built successfully

============================================================
DOCKER TEST SUMMARY
============================================================
api                  ✅ PASSED (Build: ✅)
featurizer           ✅ PASSED (Build: ✅)
parser               ✅ PASSED (Build: ✅)
trainer              ✅ PASSED (Build: ✅)
trainer-bienc        ✅ PASSED (Build: ✅)
ui                   ✅ PASSED (Build: ✅)

Results: 6/6 services passed
🎉 All Docker tests passed!

🧹 Cleaning up test images...
```

## Troubleshooting

### Common Issues

#### 1. Docker Not Available
```
❌ Docker not available or not running
Error: [Errno 2] No such file or directory: 'docker'
```
**Solution**: Install Docker and ensure it's running

#### 2. Build Timeout
```
❌ api build failed
STDERR: Command timed out after 300 seconds
```
**Solution**: Increase timeout or check network connection
```bash
python test_docker.py --timeout 600
```

#### 3. Dependency Issues
```
❌ trainer-bienc build failed
STDERR: ERROR: Could not find a version that satisfies the requirement torch==2.3.1+cpu
```
**Solution**: Check requirements.txt for version conflicts

#### 4. Permission Issues
```
❌ ui build failed
STDERR: permission denied
```
**Solution**: Ensure Docker has proper permissions

### Debug Mode

For detailed debugging, use verbose mode:
```bash
python test_docker.py --verbose --service api
```

This will show:
- Full Docker build output
- Container runtime logs
- Detailed error messages

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Docker Tests
on: [push, pull_request]

jobs:
  docker-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Docker
        run: |
          sudo apt-get update
          sudo apt-get install -y docker.io
          sudo systemctl start docker
      - name: Run Docker Tests
        run: |
          cd AI
          python test_docker.py --runtime
```

### Exit Codes
- `0`: All tests passed
- `1`: One or more tests failed

## Manual Testing

### Build Individual Service
```bash
cd AI/services/api
docker build -t ai-api .
```

### Run Individual Service
```bash
# API service
docker run -p 8000:8000 ai-api

# UI service
docker run -p 8501:8501 ai-ui

# Parser service
docker run ai-parser python parse_guidelines.py --help
```

### Test with Docker Compose
```bash
cd AI
docker-compose build  # Build all services
docker-compose up api  # Run specific service
```

## Best Practices

### 1. Regular Testing
- Run Docker tests before committing changes
- Test both build and runtime for critical services
- Use CI/CD for automated testing

### 2. Performance Optimization
- Use `.dockerignore` files to reduce build context
- Optimize layer caching in Dockerfiles
- Consider multi-stage builds for complex services

### 3. Security
- Use specific base image versions
- Scan images for vulnerabilities
- Run containers with minimal privileges

### 4. Maintenance
- Keep base images updated
- Regularly review and update dependencies
- Monitor image sizes and build times

## Service-Specific Notes

### API Service
- Exposes port 8000
- Requires model artifacts to run
- Uses FastAPI with uvicorn

### UI Service
- Exposes port 8501
- Depends on API service
- Uses Streamlit framework

### Trainer Services
- Resource-intensive builds
- May require longer timeouts
- Use CPU-only PyTorch for smaller images

### Parser Service
- Lightweight service
- Processes JSON guideline files
- Generates interim data

## Advanced Usage

### Custom Test Commands
Modify `test_commands` in `test_docker.py` to customize runtime tests:

```python
test_commands = {
    "api": ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
    "ui": ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"],
    # Add custom commands here
}
```

### Parallel Testing
For faster testing, you can run multiple services in parallel:
```bash
# Terminal 1
python test_docker.py --service api

# Terminal 2
python test_docker.py --service ui
```

### Continuous Monitoring
Set up automated testing with monitoring:
```bash
# Run tests every hour
while true; do
    python test_docker.py --runtime
    sleep 3600
done
```

## Support

For issues with Docker testing:
1. Check the troubleshooting section above
2. Run with `--verbose` for detailed output
3. Review service-specific Dockerfiles
4. Check Docker and system logs

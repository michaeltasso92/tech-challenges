#!/usr/bin/env python3
"""
Docker testing script for AI microservices
Tests Docker builds and optionally runs containers to verify they work
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

class DockerTester:
    def __init__(self, verbose: bool = False, timeout: int = 300):
        self.verbose = verbose
        self.timeout = timeout
        self.results: Dict[str, Dict] = {}
        
    def run_command(self, cmd: List[str], cwd: str = None, capture: bool = True) -> Tuple[int, str, str]:
        """Run a command and return (returncode, stdout, stderr)"""
        try:
            result = subprocess.run(
                cmd, 
                cwd=cwd,
                capture_output=capture,
                text=True,
                timeout=self.timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {self.timeout} seconds"
        except Exception as e:
            return -1, "", str(e)
    
    def check_docker_available(self) -> bool:
        """Check if Docker is available and running"""
        print("🔍 Checking Docker availability...")
        returncode, stdout, stderr = self.run_command(["docker", "--version"])
        if returncode != 0:
            print("❌ Docker not available or not running")
            print(f"Error: {stderr}")
            return False
        
        print(f"✅ Docker available: {stdout.strip()}")
        return True
    
    def get_services(self) -> List[str]:
        """Get list of services that have Dockerfiles"""
        services = []
        services_dir = Path("services")
        
        for service_dir in services_dir.iterdir():
            if service_dir.is_dir() and (service_dir / "Dockerfile").exists():
                services.append(service_dir.name)
        
        return sorted(services)
    
    def build_service(self, service: str) -> bool:
        """Build Docker image for a service"""
        print(f"\n🔨 Building {service}...")
        
        service_path = f"services/{service}"
        image_name = f"ai-{service}"
        
        # Build the image
        build_cmd = [
            "docker", "build", 
            "-t", image_name,
            "."
        ]
        
        returncode, stdout, stderr = self.run_command(build_cmd, cwd=service_path)
        
        if returncode == 0:
            print(f"✅ {service} built successfully")
            if self.verbose:
                print(stdout)
            return True
        else:
            print(f"❌ {service} build failed")
            if self.verbose:
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
            return False
    
    def test_service_runtime(self, service: str) -> bool:
        """Test if a service can start and run briefly"""
        print(f"🧪 Testing {service} runtime...")
        
        image_name = f"ai-{service}"
        container_name = f"test-{service}-{int(time.time())}"
        
        # Determine the appropriate test command based on service type
        test_commands = {
            "api": ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"],
            "ui": ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"],
            "parser": ["python", "parse_guidelines.py", "--help"],
            "trainer": ["python", "train_model.py", "--help"],
            "trainer-bienc": ["python", "train_bi.py", "--help"],
            "featurizer": ["python", "featurize_text.py", "--help"]
        }
        
        cmd = test_commands.get(service, ["python", "--version"])
        
        # Run container with a timeout
        run_cmd = [
            "docker", "run", 
            "--rm",
            "--name", container_name,
            "--network", "none",  # Isolated network for testing
            image_name
        ] + cmd
        
        returncode, stdout, stderr = self.run_command(run_cmd, timeout=30)
        
        # Clean up any remaining container
        self.run_command(["docker", "rm", "-f", container_name], capture=False)
        
        if returncode == 0:
            print(f"✅ {service} runtime test passed")
            return True
        else:
            print(f"❌ {service} runtime test failed")
            if self.verbose:
                print("STDOUT:", stdout)
                print("STDERR:", stderr)
            return False
    
    def test_service(self, service: str, test_runtime: bool = False) -> Dict:
        """Test a single service"""
        result = {
            "service": service,
            "build_success": False,
            "runtime_success": False,
            "error": None
        }
        
        try:
            # Test build
            result["build_success"] = self.build_service(service)
            
            # Test runtime if build succeeded and runtime testing is requested
            if result["build_success"] and test_runtime:
                result["runtime_success"] = self.test_service_runtime(service)
            
        except Exception as e:
            result["error"] = str(e)
        
        self.results[service] = result
        return result
    
    def cleanup_images(self, services: List[str]):
        """Clean up test images"""
        print("\n🧹 Cleaning up test images...")
        for service in services:
            image_name = f"ai-{service}"
            self.run_command(["docker", "rmi", "-f", image_name], capture=False)
    
    def run_tests(self, services: List[str], test_runtime: bool = False, cleanup: bool = True):
        """Run tests for all specified services"""
        print("🐳 Testing Docker builds for AI microservices...")
        
        if not self.check_docker_available():
            return False
        
        print(f"\n📋 Testing services: {', '.join(services)}")
        
        for service in services:
            self.test_service(service, test_runtime)
        
        # Print summary
        self.print_summary(test_runtime)
        
        # Cleanup if requested
        if cleanup:
            self.cleanup_images(services)
        
        return self.all_tests_passed(test_runtime)
    
    def print_summary(self, test_runtime: bool = False):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("DOCKER TEST SUMMARY")
        print(f"{'='*60}")
        
        passed = 0
        total = len(self.results)
        
        for service, result in self.results.items():
            build_status = "✅" if result["build_success"] else "❌"
            runtime_status = "✅" if result.get("runtime_success", True) else "❌"
            
            if test_runtime:
                status = "✅ PASSED" if result["build_success"] and result.get("runtime_success", False) else "❌ FAILED"
                print(f"{service:<20} {status} (Build: {build_status}, Runtime: {runtime_status})")
            else:
                status = "✅ PASSED" if result["build_success"] else "❌ FAILED"
                print(f"{service:<20} {status} (Build: {build_status})")
            
            if result["build_success"] and (not test_runtime or result.get("runtime_success", False)):
                passed += 1
        
        print(f"\nResults: {passed}/{total} services passed")
        
        if passed == total:
            print("🎉 All Docker tests passed!")
        else:
            print("💥 Some Docker tests failed!")
    
    def all_tests_passed(self, test_runtime: bool = False) -> bool:
        """Check if all tests passed"""
        for result in self.results.values():
            if not result["build_success"]:
                return False
            if test_runtime and not result.get("runtime_success", False):
                return False
        return True

def main():
    parser = argparse.ArgumentParser(description="Test Docker builds for AI microservices")
    parser.add_argument("--service", help="Test specific service only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--runtime", action="store_true", help="Test runtime execution (slower)")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up test images")
    parser.add_argument("--timeout", type=int, default=300, help="Build timeout in seconds")
    
    args = parser.parse_args()
    
    # Change to AI directory
    ai_dir = Path(__file__).parent
    os.chdir(ai_dir)
    
    tester = DockerTester(verbose=args.verbose, timeout=args.timeout)
    
    # Get available services
    all_services = tester.get_services()
    
    if not all_services:
        print("❌ No services with Dockerfiles found")
        sys.exit(1)
    
    # Determine which services to test
    if args.service:
        if args.service in all_services:
            services_to_test = [args.service]
        else:
            print(f"❌ Service '{args.service}' not found. Available services: {', '.join(all_services)}")
            sys.exit(1)
    else:
        services_to_test = all_services
    
    # Run tests
    success = tester.run_tests(
        services_to_test, 
        test_runtime=args.runtime,
        cleanup=not args.no_cleanup
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

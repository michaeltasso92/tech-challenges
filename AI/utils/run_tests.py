#!/usr/bin/env python3
"""
Test runner for all AI microservices
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_tests(service_path, verbose=False):
    """Run tests for a specific service"""
    print(f"\n{'='*60}")
    print(f"Running tests for {service_path}")
    print(f"{'='*60}")
    
    # Change to service directory
    original_dir = os.getcwd()
    os.chdir(service_path)
    
    try:
        # Install requirements (now includes test dependencies)
        req_file = "requirements.txt"
        if os.path.exists(req_file):
            print(f"Installing requirements from {req_file}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], 
                         check=True, capture_output=not verbose)
        
        # Run pytest
        cmd = [sys.executable, "-m", "pytest", "tests/", "-v" if verbose else ""]
        cmd = [arg for arg in cmd if arg]  # Remove empty strings
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=not verbose, text=True)
        
        if result.returncode == 0:
            print(f"✅ Tests passed for {service_path}")
            return True
        else:
            print(f"❌ Tests failed for {service_path}")
            if not verbose:
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running tests for {service_path}: {e}")
        return False
    finally:
        os.chdir(original_dir)

def main():
    parser = argparse.ArgumentParser(description="Run tests for all AI microservices")
    parser.add_argument("--service", help="Run tests for specific service only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")
    
    args = parser.parse_args()
    
    # Define services
    services = [
        "services/api",
        "services/parser", 
        "services/trainer",
        "services/trainer-bienc",
        "services/ui"
    ]
    
    # Change to AI directory
    ai_dir = Path(__file__).parent
    os.chdir(ai_dir)
    
    if args.install_deps:
        print("Installing dependencies for all services...")
        for service in services:
            req_file = os.path.join(service, "requirements.txt")
            if os.path.exists(req_file):
                print(f"Installing dependencies for {service}...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_file], 
                             check=True)
        print("✅ All dependencies installed")
        return
    
    # Run tests
    if args.service:
        if args.service in services:
            success = run_tests(args.service, args.verbose)
            sys.exit(0 if success else 1)
        else:
            print(f"❌ Service '{args.service}' not found. Available services: {', '.join(services)}")
            sys.exit(1)
    
    # Run all tests
    print("🧪 Running tests for all AI microservices...")
    
    results = {}
    for service in services:
        if os.path.exists(os.path.join(service, "tests")):
            results[service] = run_tests(service, args.verbose)
        else:
            print(f"⚠️  No tests found for {service}")
            results[service] = True  # Skip if no tests
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for service, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{service:<20} {status}")
        if success:
            passed += 1
    
    print(f"\nResults: {passed}/{total} services passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("💥 Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

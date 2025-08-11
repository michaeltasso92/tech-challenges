#!/usr/bin/env python3
"""
Quick script to check Docker status of all AI microservices
"""
import os
from pathlib import Path

def check_docker_status():
    """Check which services have Dockerfiles and their status"""
    print("🐳 Docker Status Check for AI Microservices")
    print("=" * 50)
    
    services_dir = Path("services")
    services_with_docker = []
    services_without_docker = []
    
    for service_dir in services_dir.iterdir():
        if service_dir.is_dir():
            service_name = service_dir.name
            dockerfile_path = service_dir / "Dockerfile"
            requirements_path = service_dir / "requirements.txt"
            
            if dockerfile_path.exists():
                services_with_docker.append({
                    "name": service_name,
                    "dockerfile": True,
                    "requirements": requirements_path.exists(),
                    "size": dockerfile_path.stat().st_size
                })
            else:
                services_without_docker.append(service_name)
    
    # Print services with Docker
    print(f"\n✅ Services with Dockerfiles ({len(services_with_docker)}):")
    for service in services_with_docker:
        req_status = "✅" if service["requirements"] else "❌"
        print(f"  • {service['name']:<15} {req_status} requirements.txt")
    
    # Print services without Docker
    if services_without_docker:
        print(f"\n❌ Services without Dockerfiles ({len(services_without_docker)}):")
        for service in services_without_docker:
            print(f"  • {service}")
    
    # Summary
    total_services = len(services_with_docker) + len(services_without_docker)
    docker_coverage = (len(services_with_docker) / total_services) * 100 if total_services > 0 else 0
    
    print(f"\n📊 Summary:")
    print(f"  Total services: {total_services}")
    print(f"  Dockerized: {len(services_with_docker)}")
    print(f"  Coverage: {docker_coverage:.1f}%")
    
    # Check docker-compose.yml
    compose_path = Path("docker-compose.yml")
    if compose_path.exists():
        print(f"\n✅ docker-compose.yml found")
        # Count services in docker-compose
        with open(compose_path) as f:
            content = f.read()
            compose_services = [line.strip() for line in content.split('\n') 
                              if line.strip().startswith('  ') and ':' in line and not line.strip().startswith('    ')]
        print(f"  Services in compose: {len(compose_services)}")
    else:
        print(f"\n❌ docker-compose.yml not found")
    
    return len(services_with_docker), len(services_without_docker)

if __name__ == "__main__":
    check_docker_status()

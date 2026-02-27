#!/usr/bin/env python3
"""
Production Readiness Verification Script
Checks all Docker and application configuration for production deployment
"""

import os
import sys
import json
from pathlib import Path

class ProductionChecker:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        
    def check(self, name, condition, details=""):
        """Run a check and log result"""
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"{status} | {name}")
        if details and not condition:
            print(f"      └─ {details}")
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            
    def section(self, title):
        """Print section header"""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def summary(self):
        """Print final summary"""
        total = self.passed + self.failed
        percent = (self.passed / total * 100) if total > 0 else 0
        print(f"\n{'='*60}")
        print(f"  SUMMARY: {self.passed}/{total} checks passed ({percent:.0f}%)")
        print(f"{'='*60}\n")
        return self.failed == 0

def main():
    checker = ProductionChecker()
    root = Path(__file__).parent
    
    # ===== Docker Configuration =====
    checker.section("DOCKER CONFIGURATION")
    
    dockerfile = root / "Dockerfile"
    checker.check("Dockerfile exists", dockerfile.exists(), str(dockerfile))
    
    if dockerfile.exists():
        content = dockerfile.read_text(encoding='utf-8', errors='ignore')
        checker.check("Dockerfile: Multi-stage build", "AS builder" in content, "Missing 'AS builder' stage")
        checker.check("Dockerfile: Non-root user", "appuser" in content, "Not running as non-root user")
        checker.check("Dockerfile: Security labels", "LABEL" in content, "Missing security labels")
        checker.check("Dockerfile: Health check", "HEALTHCHECK" in content, "Missing health check")
        checker.check("Dockerfile: Signal handling", "STOPSIGNAL" in content, "Missing STOPSIGNAL")
        checker.check("Dockerfile: Pre-built PyTorch", "index-url https://download.pytorch.org/whl/cpu" in content or "torch==2.3.1" in content, "PyTorch not specified or not CPU-optimized")
    
    # ===== Docker Compose =====
    checker.section("DOCKER-COMPOSE CONFIGURATION")
    
    compose = root / "docker-compose.yml"
    checker.check("docker-compose.yml exists", compose.exists(), str(compose))
    
    if compose.exists():
        content = compose.read_text(encoding='utf-8', errors='ignore')
        checker.check("Compose: Restart policy", "restart:" in content, "No restart policy defined")
        checker.check("Compose: Resource limits", "limits:" in content, "No resource limits defined")
        checker.check("Compose: Health checks", "healthcheck:" in content, "No health checks defined")
        checker.check("Compose: Volumes", "volumes:" in content, "No volumes for data persistence")
        checker.check("Compose: Logging config", "logging:" in content, "No logging configuration")
        checker.check("Compose: Environment vars", "environment:" in content, "No environment variables")
        checker.check("Compose: Security options", "security_opt:" in content, "No security options")
    
    # ===== Docker Ignore =====
    checker.section(".DOCKERIGNORE OPTIMIZATION")
    
    dockerignore = root / ".dockerignore"
    checker.check(".dockerignore exists", dockerignore.exists(), str(dockerignore))
    
    if dockerignore.exists():
        content = dockerignore.read_text(encoding='utf-8', errors='ignore')
        lines = len([l for l in content.split('\n') if l.strip() and not l.startswith('#')])
        checker.check(".dockerignore: Comprehensive rules", lines > 30, f"Only {lines} rules (recommended >30)")
        checker.check(".dockerignore: venv ignored", "venv/" in content, "Virtual environment not excluded")
        checker.check(".dockerignore: .git ignored", ".git" in content, "Git directory not excluded")
        checker.check(".dockerignore: IDE ignored", ".vscode/" in content and ".idea/" in content, "IDE directories not excluded")
    
    # ===== Requirements & Environment =====
    checker.section("PYTHON & DEPENDENCIES")
    
    requirements = root / "requirements.txt"
    checker.check("requirements.txt exists", requirements.exists(), str(requirements))
    
    if requirements.exists():
        content = requirements.read_text(encoding='utf-8', errors='ignore')
        checker.check("requirements.txt: torch", "torch" in content, "PyTorch not specified")
        checker.check("requirements.txt: pandas", "pandas" in content, "Pandas not specified")
        checker.check("requirements.txt: prophet", "prophet" in content, "Prophet not specified")
        checker.check("requirements.txt: optuna", "optuna" in content, "Optuna not specified")
        checker.check("requirements.txt: pytest", "pytest" in content, "Pytest not specified (needed for testing)")
    
    env_file = root / ".env"
    checker.check(".env file exists", env_file.exists(), str(env_file))
    
    # ===== Source Code =====
    checker.section("SOURCE CODE STRUCTURE")
    
    src_dir = root / "src"
    checker.check("src/ directory exists", src_dir.exists(), str(src_dir))
    
    if src_dir.exists():
        train_py = src_dir / "train.py"
        preprocess_py = src_dir / "preprocess.py"
        models_py = src_dir / "models.py"
        feature_eng_py = src_dir / "feature_engineering.py"
        
        checker.check("src/train.py exists", train_py.exists(), str(train_py))
        checker.check("src/preprocess.py exists", preprocess_py.exists(), str(preprocess_py))
        checker.check("src/models.py exists", models_py.exists(), str(models_py))
        checker.check("src/feature_engineering.py exists", feature_eng_py.exists(), str(feature_eng_py))
        
        if train_py.exists():
            content = train_py.read_text(encoding='utf-8', errors='ignore')
            checker.check("train.py: Error handling", "try:" in content and "except" in content, "Missing try-except blocks")
            checker.check("train.py: Logging", "LOG_PATH" in content, "Missing logging configuration")
            checker.check("train.py: Result paths", "RESULT_PATH" in content, "Missing results directory handling")
    
    # ===== Data & Output Directories =====
    checker.section("DATA & OUTPUT DIRECTORIES")
    
    data_dir = root / "data"
    results_dir = root / "results"
    logs_dir = root / "logs"
    models_dir = root / "models"
    
    checker.check("data/ directory exists", data_dir.exists(), "Create with: mkdir data/")
    checker.check("results/ directory exists", results_dir.exists(), "Create with: mkdir results/")
    checker.check("logs/ directory exists", logs_dir.exists(), "Create with: mkdir logs/")
    checker.check("models/ directory exists", models_dir.exists(), "Create with: mkdir models/")
    
    # ===== Tests =====
    checker.section("TESTING SETUP")
    
    tests_dir = root / "tests"
    checker.check("tests/ directory exists", tests_dir.exists(), str(tests_dir))
    
    if tests_dir.exists():
        test_files = list(tests_dir.glob("test_*.py"))
        checker.check("Test files exist", len(test_files) > 0, f"Found {len(test_files)} test files")
    
    # ===== Documentation =====
    checker.section("DOCUMENTATION")
    
    readme = root / "README.md"
    docker_guide = root / "DOCKER_SETUP_COMPLETE.md"
    
    checker.check("README.md exists", readme.exists(), "Create project documentation")
    checker.check("DOCKER_SETUP_COMPLETE.md exists", docker_guide.exists(), "Docker setup guide")
    
    # ===== Git Configuration =====
    checker.section("GIT & VERSION CONTROL")
    
    git_dir = root / ".git"
    gitignore = root / ".gitignore"
    
    checker.check(".git directory exists", git_dir.exists(), "Initialize with: git init")
    checker.check(".gitignore exists", gitignore.exists(), "Create .gitignore file")
    
    # Final Summary
    success = checker.summary()
    
    if success:
        print("🎉 ALL CHECKS PASSED! Ready for production deployment!")
        print("\nNext steps:")
        print("  1. docker builder prune -a -f")
        print("  2. docker-compose up --build")
        print("  3. Monitor with: docker-compose logs -f")
        return 0
    else:
        print("⚠️  SOME CHECKS FAILED! Please review the issues above.")
        print("\nTo fix missing directories:")
        print("  mkdir -p data/raw data/processed results logs models")
        print("\nTo fix missing Docker configs:")
        print("  - Review Dockerfile for multi-stage build")
        print("  - Review docker-compose.yml for resource limits & health checks")
        print("  - Review .dockerignore for optimization rules")
        return 1

if __name__ == "__main__":
    sys.exit(main())

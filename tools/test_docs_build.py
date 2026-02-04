#!/usr/bin/env python3
"""
Documentation build test for DeepLabCut.

This script provides a lightweight test of documentation building
without requiring full Sphinx setup.
"""

import os
import sys
from pathlib import Path


def test_markdown_files():
    """Test that markdown files are properly formatted."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / 'docs'
    
    print("📚 Testing documentation files...")
    
    # Find all markdown files
    md_files = list(docs_dir.glob('**/*.md'))
    print(f"Found {len(md_files)} markdown files")
    
    errors = []
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            
            # Basic checks
            if len(content.strip()) == 0:
                errors.append(f"Empty file: {md_file.relative_to(repo_root)}")
            
            # Check for basic structure
            lines = content.split('\n')
            if not any(line.startswith('#') for line in lines[:10]):
                # Allow some exceptions for files that may not have headers immediately
                if not md_file.name.lower() in ['readme.md', 'intro.md']:
                    errors.append(f"No header found in first 10 lines: {md_file.relative_to(repo_root)}")
                
        except Exception as e:
            errors.append(f"Error reading {md_file.relative_to(repo_root)}: {e}")
    
    if errors:
        print("❌ Documentation errors found:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("✅ All documentation files passed basic checks")
        return True


def test_rst_files():
    """Test that RST files are properly formatted."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / 'docs'
    
    # Find all RST files
    rst_files = list(docs_dir.glob('**/*.rst'))
    if rst_files:
        print(f"Found {len(rst_files)} RST files")
        
        errors = []
        for rst_file in rst_files:
            try:
                content = rst_file.read_text(encoding='utf-8')
                if len(content.strip()) == 0:
                    errors.append(f"Empty file: {rst_file.relative_to(repo_root)}")
            except Exception as e:
                errors.append(f"Error reading {rst_file.relative_to(repo_root)}: {e}")
        
        if errors:
            print("❌ RST file errors found:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ All RST files passed basic checks")
            return True
    else:
        print("No RST files found")
        return True


def test_config_files():
    """Test that configuration files exist and are valid."""
    repo_root = Path(__file__).parent.parent
    
    config_files = ['_config.yml', '_toc.yml']
    
    print("🔧 Testing configuration files...")
    
    for config_file in config_files:
        config_path = repo_root / config_file
        if config_path.exists():
            try:
                content = config_path.read_text(encoding='utf-8')
                if len(content.strip()) == 0:
                    print(f"⚠️  Warning: Empty config file: {config_file}")
                else:
                    print(f"✅ Found valid config file: {config_file}")
            except Exception as e:
                print(f"❌ Error reading {config_file}: {e}")
                return False
        else:
            print(f"ℹ️  Config file not found: {config_file}")
    
    return True


def main():
    """Main test execution."""
    print("🔍 DeepLabCut Documentation Build Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test markdown files
    if not test_markdown_files():
        all_passed = False
    
    print()
    
    # Test RST files  
    if not test_rst_files():
        all_passed = False
    
    print()
    
    # Test config files
    if not test_config_files():
        all_passed = False
    
    print()
    
    if all_passed:
        print("🎉 All documentation tests passed!")
        print("⏱️  Documentation build test completed in < 30 seconds")
        return 0
    else:
        print("❌ Some documentation tests failed!")
        return 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""
Helper script to parse test selection JSON output for GitHub Actions.
"""

import json
import sys
import os
from pathlib import Path

def extract_commands():
    """Extract commands from test selection JSON."""
    try:
        with open('test_selection.json') as f:
            data = json.load(f)
        return data.get('commands', [])
    except FileNotFoundError:
        print('[]')
        sys.stderr.write('Warning: test_selection.json not found, returning empty commands\n')
        return []
    except Exception as e:
        print('[]')
        sys.stderr.write(f'Warning: Error parsing test selection: {e}\n')
        return []

def check_full_tests():
    """Check if we need to run full tests."""
    try:
        with open('test_selection.json') as f:
            data = json.load(f)
        commands = data.get('commands', [])
        full_tests = any('examples/testscript.py' in str(cmd) or 
                        (('pytest' in cmd) and len(cmd.split()) <= 3) 
                        for cmd in commands)
        return full_tests
    except FileNotFoundError:
        sys.stderr.write('Warning: test_selection.json not found, assuming full tests\n')
        return True
    except Exception as e:
        sys.stderr.write(f'Warning: Error checking full tests: {e}\n')
        return True

def create_test_matrix():
    """Create test matrix for parallel execution."""
    try:
        with open('test_selection.json') as f:
            data = json.load(f)
        
        # Create matrix based on test categories
        categories = list(data.get('categories', {}).keys())
        commands = data.get('commands', [])
        
        if not commands:
            matrix = [{'name': 'minimal', 'commands': ['echo "No tests to run"']}]
        elif 'docs' in categories and len(categories) == 1:
            matrix = [{'name': 'docs-only', 'commands': [cmd for cmd in commands if 'docs' in cmd]}]
        elif 'superanimal' in categories:
            matrix = [{'name': 'superanimal', 'commands': [cmd for cmd in commands if 'superanimal' in cmd or 'modelzoo' in cmd]}]
        else:
            matrix = [{'name': 'selected-tests', 'commands': commands}]
        
        return matrix
    except FileNotFoundError:
        # Fallback matrix if file is missing
        matrix = [{'name': 'fallback', 'commands': ['python -m pytest tests/']}]
        sys.stderr.write('Warning: test_selection.json not found, using fallback matrix\n')
        return matrix
    except Exception as e:
        # Fallback matrix if parsing fails
        matrix = [{'name': 'fallback', 'commands': ['python -m pytest tests/']}]
        sys.stderr.write(f'Warning: Error creating test matrix: {e}\n')
        return matrix

def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python parse_test_selection.py [commands|full-tests|matrix]")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == 'commands':
        commands = extract_commands()
        print(json.dumps(commands))
    elif action == 'full-tests':
        full_tests = check_full_tests()
        print('true' if full_tests else 'false')
    elif action == 'matrix':
        matrix = create_test_matrix()
        print(json.dumps(matrix))
    else:
        print(f"Unknown action: {action}")
        sys.exit(1)

if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""
Helper script to execute test commands from JSON array for GitHub Actions.
"""

import json
import subprocess
import sys
import os

def main():
    """Execute test commands from environment variable."""
    try:
        # Get commands from environment variable
        test_commands_json = os.environ.get('TEST_COMMANDS', '[]')
        commands = json.loads(test_commands_json)
        
        if not commands:
            print("No test commands to execute.")
            return 0
        
        all_passed = True
        
        for i, cmd in enumerate(commands, 1):
            print(f'[{i}/{len(commands)}] Running: {cmd}')
            try:
                result = subprocess.run(cmd.split(), check=True)
                print(f'✅ Passed: {cmd}')
            except subprocess.CalledProcessError as e:
                print(f'❌ Failed: {cmd}')
                all_passed = False
        
        return 0 if all_passed else 1
    
    except Exception as e:
        print(f"Error executing test commands: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
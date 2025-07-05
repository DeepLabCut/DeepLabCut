#!/usr/bin/env python3
"""
Intelligent test selection system for DeepLabCut.

This script analyzes git diff to determine which tests should be run based on the
files that have been changed in a pull request. It aims to reduce CI runtime to
approximately 5 minutes by running only relevant tests.

Categories:
1. Documentation-only changes -> run doc build tests only
2. SuperAnimal model changes -> run SuperAnimal tests only  
3. Python script changes -> run pytest for those specific scripts
4. Complex changes -> run full test suite
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class TestSelector:
    """Intelligent test selector based on git diff analysis."""
    
    def __init__(self, repo_root: str = None):
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        
        # Define file patterns and their corresponding test categories
        self.test_mappings = {
            'docs': {
                'patterns': ['docs/', '*.md', '*.rst', '_config.yml', '_toc.yml'],
                'tests': ['docs'],
                'commands': ['python tools/test_docs_build.py']  # Build documentation
            },
            'superanimal': {
                'patterns': [
                    'deeplabcut/pose_estimation_tensorflow/modelzoo/api/superanimal',
                    'deeplabcut/pose_estimation_pytorch/modelzoo',
                    'deeplabcut/modelzoo',
                    'deeplabcut/create_project/modelzoo.py',
                    'deeplabcut/gui/tabs/modelzoo.py',
                    'deeplabcut/*superanimal*',
                    'deeplabcut/*modelzoo*'
                ],
                'tests': [
                    'tests/test_predict_supermodel.py',
                    'tests/pose_estimation_pytorch/modelzoo/',
                    'tests/pose_estimation_pytorch/other/test_modelzoo.py'
                ],
                'commands': []
            },
            'core': {
                'patterns': [
                    'deeplabcut/core/',
                    'deeplabcut/pose_estimation_tensorflow/',
                    'deeplabcut/pose_estimation_pytorch/',
                    'deeplabcut/utils/',
                    'deeplabcut/auxiliaryfunctions.py'
                ],
                'tests': [
                    'tests/test_auxiliaryfunctions.py',
                    'tests/core/',
                    'tests/pose_estimation_pytorch/',
                    'tests/utils/'
                ],
                'commands': []
            },
            'multianimal': {
                'patterns': [
                    'deeplabcut/*multianimal*',
                    'deeplabcut/pose_estimation_tensorflow/*multi*',
                    'deeplabcut/pose_estimation_pytorch/*multi*'
                ],
                'tests': [
                    'tests/test_auxfun_multianimal.py',
                    'tests/test_pose_multianimal_imgaug.py',
                    'tests/test_predict_multianimal.py',
                    'examples/testscript_multianimal.py'
                ],
                'commands': []
            },
            'video': {
                'patterns': [
                    'deeplabcut/*video*',
                    'deeplabcut/pose_estimation_tensorflow/nnet/predict.py',
                    'deeplabcut/pose_estimation_pytorch/apis/videos.py'
                ],
                'tests': [
                    'tests/test_video.py'
                ],
                'commands': []
            },
            'tools': {
                'patterns': ['tools/'],
                'tests': [],
                'commands': []
            }
        }
        
    def get_changed_files(self, base_ref: str = 'origin/main') -> List[str]:
        """Get list of files changed compared to base reference."""
        try:
            # First fetch the base ref to ensure it exists
            try:
                subprocess.run(
                    ['git', 'fetch', 'origin', base_ref.replace('origin/', '')],
                    cwd=self.repo_root,
                    capture_output=True,
                    check=False
                )
            except:
                pass  # Ignore fetch errors
            
            # Try multiple approaches to get changed files
            approaches = [
                # Compare with merge base
                lambda: self._get_files_from_merge_base(base_ref),
                # Compare directly with base ref
                lambda: self._get_files_direct_diff(base_ref),
                # Get staged changes
                lambda: self._get_staged_files(),
                # Get working directory changes
                lambda: self._get_working_dir_files()
            ]
            
            for approach in approaches:
                try:
                    changed_files = approach()
                    if changed_files:
                        return changed_files
                except subprocess.CalledProcessError:
                    continue
                    
            return []
            
        except Exception as e:
            print(f"Warning: Could not get changed files: {e}")
            return []
    
    def _get_files_from_merge_base(self, base_ref: str) -> List[str]:
        """Get files changed since merge base."""
        merge_base = subprocess.check_output(
            ['git', 'merge-base', 'HEAD', base_ref],
            cwd=self.repo_root,
            text=True
        ).strip()
        
        result = subprocess.check_output(
            ['git', 'diff', '--name-only', merge_base, 'HEAD'],
            cwd=self.repo_root,
            text=True
        )
        
        return [f.strip() for f in result.split('\n') if f.strip()]
    
    def _get_files_direct_diff(self, base_ref: str) -> List[str]:
        """Get files changed directly compared to base ref."""
        result = subprocess.check_output(
            ['git', 'diff', '--name-only', base_ref, 'HEAD'],
            cwd=self.repo_root,
            text=True
        )
        
        return [f.strip() for f in result.split('\n') if f.strip()]
    
    def _get_staged_files(self) -> List[str]:
        """Get staged files."""
        result = subprocess.check_output(
            ['git', 'diff', '--name-only', '--cached'],
            cwd=self.repo_root,
            text=True
        )
        
        return [f.strip() for f in result.split('\n') if f.strip()]
    
    def _get_working_dir_files(self) -> List[str]:
        """Get working directory changes."""
        result = subprocess.check_output(
            ['git', 'diff', '--name-only'],
            cwd=self.repo_root,
            text=True
        )
        
        return [f.strip() for f in result.split('\n') if f.strip()]
    
    def match_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any of the given patterns."""
        import fnmatch
        
        for pattern in patterns:
            # Direct path matching
            if pattern in file_path:
                return True
            # Glob pattern matching (handle wildcards)
            if fnmatch.fnmatch(file_path, pattern):
                return True
            # Directory matching
            if pattern.endswith('/') and file_path.startswith(pattern):
                return True
            # Handle wildcards in path components
            if '*' in pattern:
                pattern_parts = pattern.split('/')
                file_parts = file_path.split('/')
                if self._match_path_components(file_parts, pattern_parts):
                    return True
                
        return False
    
    def _match_path_components(self, file_parts: List[str], pattern_parts: List[str]) -> bool:
        """Match file path components against pattern components with wildcards."""
        import fnmatch
        
        # Simple case: direct fnmatch
        if len(pattern_parts) == len(file_parts):
            return all(fnmatch.fnmatch(f, p) for f, p in zip(file_parts, pattern_parts))
        
        # Handle wildcards in any part of the path
        for i, pattern_part in enumerate(pattern_parts):
            if '*' in pattern_part:
                for j, file_part in enumerate(file_parts):
                    if fnmatch.fnmatch(file_part, pattern_part):
                        return True
        
        return False
    
    def categorize_changes(self, changed_files: List[str]) -> Dict[str, List[str]]:
        """Categorize changed files into test categories."""
        categories = {}
        uncategorized = []
        
        # Priority order for categorization (higher priority first)
        priority_order = ['superanimal', 'multianimal', 'core', 'video', 'docs', 'tools']
        
        for file_path in changed_files:
            matched = False
            # Check categories in priority order
            for category in priority_order:
                if category in self.test_mappings:
                    config = self.test_mappings[category]
                    if self.match_patterns(file_path, config['patterns']):
                        if category not in categories:
                            categories[category] = []
                        categories[category].append(file_path)
                        matched = True
                        break  # Take first match in priority order
            
            if not matched:
                uncategorized.append(file_path)
        
        if uncategorized:
            categories['uncategorized'] = uncategorized
            
        return categories
    
    def get_tests_to_run(self, categories: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
        """Determine which tests and commands to run based on categorized changes."""
        tests_to_run = set()
        commands_to_run = set()
        
        # If only documentation changes, just build docs
        if len(categories) == 1 and 'docs' in categories:
            commands_to_run.update(self.test_mappings['docs']['commands'])
            return list(tests_to_run), list(commands_to_run)
        
        # If SuperAnimal changes and limited other changes, focus on SuperAnimal
        if 'superanimal' in categories:
            superanimal_files = len(categories.get('superanimal', []))
            total_files = sum(len(files) for files in categories.values())
            
            # If SuperAnimal changes dominate or are the main focus
            if superanimal_files >= total_files // 2 or len(categories) <= 2:
                config = self.test_mappings['superanimal']
                tests_to_run.update(config['tests'])
                commands_to_run.update(config['commands'])
                
                # Add docs tests if docs also changed
                if 'docs' in categories:
                    commands_to_run.update(self.test_mappings['docs']['commands'])
                
                return list(tests_to_run), list(commands_to_run)
        
        # If uncategorized files or too many mixed categories, run full test suite
        if 'uncategorized' in categories or len(categories) > 3:
            return ['pytest'], ['python examples/testscript.py']
        
        # For other focused changes, run specific tests
        focused_categories = ['multianimal', 'core', 'video']
        for category in focused_categories:
            if category in categories and len(categories) <= 2:
                config = self.test_mappings[category]
                tests_to_run.update(config['tests'])
                commands_to_run.update(config['commands'])
                
                # Add docs tests if docs also changed
                if 'docs' in categories:
                    commands_to_run.update(self.test_mappings['docs']['commands'])
                
                return list(tests_to_run), list(commands_to_run)
        
        # Default: add tests for each category but be selective
        for category in categories:
            if category in self.test_mappings:
                config = self.test_mappings[category]
                tests_to_run.update(config['tests'])
                commands_to_run.update(config['commands'])
        
        return list(tests_to_run), list(commands_to_run)
    
    def filter_existing_paths(self, paths: List[str]) -> List[str]:
        """Filter out non-existent test paths."""
        existing_paths = []
        for path in paths:
            full_path = self.repo_root / path
            if full_path.exists():
                existing_paths.append(path)
            elif path == 'pytest':  # Special case for pytest command
                existing_paths.append(path)
            else:
                print(f"Warning: Test path does not exist: {path}")
        return existing_paths
    
    def generate_test_commands(self, tests: List[str], commands: List[str]) -> List[str]:
        """Generate the actual commands to run tests."""
        all_commands = []
        
        # Add custom commands first
        all_commands.extend(commands)
        
        # Group pytest commands
        pytest_paths = [t for t in tests if t != 'pytest' and not t.startswith('examples/')]
        functional_tests = [t for t in tests if t.startswith('examples/')]
        
        if 'pytest' in tests:
            # Run full pytest suite
            all_commands.append('python -m pytest')
        elif pytest_paths:
            # Run specific pytest paths
            existing_paths = self.filter_existing_paths(pytest_paths)
            if existing_paths:
                all_commands.append(f'python -m pytest {" ".join(existing_paths)}')
        
        # Add functional test commands
        for test in functional_tests:
            if self.repo_root.joinpath(test).exists():
                all_commands.append(f'python {test}')
        
        return all_commands
    
    def run(self, base_ref: str = 'origin/main', dry_run: bool = False) -> Dict:
        """Main execution method."""
        print("🔍 DeepLabCut Intelligent Test Selector")
        print("=" * 50)
        
        # Get changed files
        changed_files = self.get_changed_files(base_ref)
        print(f"📁 Found {len(changed_files)} changed files:")
        for file in changed_files[:10]:  # Show first 10
            print(f"   - {file}")
        if len(changed_files) > 10:
            print(f"   ... and {len(changed_files) - 10} more files")
        print()
        
        if not changed_files:
            print("⚠️  No changed files detected. Running minimal test suite.")
            return {
                'changed_files': [],
                'categories': {},
                'tests': ['tests/test_auxiliaryfunctions.py'],
                'commands': ['python -m pytest tests/test_auxiliaryfunctions.py'],
                'estimated_time': '2 minutes'
            }
        
        # Categorize changes
        categories = self.categorize_changes(changed_files)
        print("📂 Change categories:")
        for category, files in categories.items():
            print(f"   - {category}: {len(files)} files")
        print()
        
        # Determine tests to run
        tests, commands = self.get_tests_to_run(categories)
        test_commands = self.generate_test_commands(tests, commands)
        
        # Estimate runtime
        estimated_time = self.estimate_runtime(categories, len(test_commands))
        
        print("🧪 Tests to run:")
        for cmd in test_commands:
            print(f"   - {cmd}")
        print(f"\n⏱️  Estimated runtime: {estimated_time}")
        
        result = {
            'changed_files': changed_files,
            'categories': categories,
            'tests': tests,
            'commands': test_commands,
            'estimated_time': estimated_time
        }
        
        if not dry_run:
            print("\n🚀 Executing tests...")
            self.execute_tests(test_commands)
        
        return result
    
    def estimate_runtime(self, categories: Dict[str, List[str]], num_commands: int) -> str:
        """Estimate test runtime based on categories and number of commands."""
        if 'docs' in categories and len(categories) == 1:
            return "1-2 minutes"
        elif 'superanimal' in categories and len(categories) <= 2:
            return "3-4 minutes" 
        elif len(categories) <= 2 and 'uncategorized' not in categories:
            return "2-3 minutes"
        elif num_commands == 1 and 'pytest' not in str(categories):
            return "1-2 minutes"
        else:
            return "5+ minutes (full test suite)"
    
    def execute_tests(self, commands: List[str]) -> bool:
        """Execute the test commands."""
        all_passed = True
        
        for i, cmd in enumerate(commands, 1):
            print(f"\n[{i}/{len(commands)}] Running: {cmd}")
            print("-" * 40)
            
            try:
                result = subprocess.run(
                    cmd.split(),
                    cwd=self.repo_root,
                    check=True,
                    capture_output=False
                )
                print(f"✅ Command passed: {cmd}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Command failed: {cmd}")
                print(f"   Exit code: {e.returncode}")
                all_passed = False
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Intelligent test selection for DeepLabCut",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/test_selector.py                    # Run tests for current changes
  python tools/test_selector.py --dry-run          # Show what tests would run
  python tools/test_selector.py --base main        # Compare against main branch
  python tools/test_selector.py --output-json      # Output results as JSON
        """
    )
    
    parser.add_argument(
        '--base', 
        default='origin/main',
        help='Base reference for git diff (default: origin/main)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Show what tests would run without executing them'
    )
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='Output results in JSON format'
    )
    parser.add_argument(
        '--repo-root',
        help='Repository root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    try:
        selector = TestSelector(args.repo_root)
        result = selector.run(args.base, args.dry_run)
        
        if args.output_json:
            import json
            print(json.dumps(result, indent=2))
        
        # Exit with appropriate code
        if args.dry_run or result['commands']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
"""Check release text and Git attribution without printing matched private text."""
from pathlib import Path
import argparse
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--forbid', action='append', default=[], help='Additional personal identifier to reject')
    args = parser.parse_args()
    root = args.root.resolve()
    patterns = {
        'personal home path': re.compile('/' + r'(?:home|Users)/[^\s/\"\x27]+'),
        'absolute Windows path': re.compile(r'\b[A-Za-z]:' + r'[\\/]'),
        'private mount path': re.compile('/' + r'(?:mnt/storage|data/users)/'),
        'email address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'),
        'private key': re.compile('BEGIN ' + r'(?:OPENSSH|RSA|EC|DSA) PRIVATE KEY'),
    }
    for index, value in enumerate(args.forbid):
        patterns['provided identifier ' + str(index + 1)] = re.compile(re.escape(value), re.I)
    git = subprocess.run(['git', '-C', str(root), 'rev-parse', '--show-toplevel'], capture_output=True, text=True)
    own_git = git.returncode == 0 and Path(git.stdout.strip()).resolve() == root
    if own_git:
        names = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z']).decode().split('\0')
        paths = [root / name for name in names if name]
    else:
        paths = [p for p in root.rglob('*') if p.is_file() and not any(
            part in {'.git', '__pycache__', '.pytest_cache'} for part in p.relative_to(root).parts)]
    findings = []
    for path in paths:
        if path.name == '.env' or path.suffix in {'.pem', '.key', '.pt', '.pth', '.npy', '.npz', '.h5', '.hdf5', '.safetensors'}:
            findings.append((path.relative_to(root).as_posix(), 0, 'non-source or credential file'))
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            findings.append((path.relative_to(root).as_posix(), 0, 'binary release file'))
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns.items():
                matches = list(pattern.finditer(line))
                if label == 'email address':
                    matches = [m for m in matches if m.group(0) != 'anonymous@example.invalid']
                if matches:
                    findings.append((path.relative_to(root).as_posix(), number, label))
    if own_git:
        history = subprocess.run(['git', '-C', str(root), 'log', '--format=%an <%ae>%n%cn <%ce>'],
                                 capture_output=True, text=True)
        if history.returncode == 0:
            if any(line != 'Anonymous <anonymous@example.invalid>' for line in history.stdout.splitlines() if line):
                findings.append(('.git', 0, 'non-anonymous commit attribution'))
    for path, line, kind in findings:
        print(f'{path}:{line}: {kind}')
    print(f'Checked {len(paths)} source files; {len(findings)} findings.')
    return bool(findings)


if __name__ == '__main__':
    sys.exit(main())

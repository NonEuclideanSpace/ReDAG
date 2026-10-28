set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f deployment/environment.sh ]; then source deployment/environment.sh; fi
if [ "$#" -eq 0 ]; then set -- --help; fi
if [ "$1" = tools ]; then
    shift
    exec .venv-tools/bin/python scripts/serve_tools.py "$@"
fi
exec .venv/bin/agentic-image "$@"

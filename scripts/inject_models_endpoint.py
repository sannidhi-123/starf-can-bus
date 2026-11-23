import sys

content = open('backend/main.py', 'r', encoding='utf-8').read()

new_endpoint = (
    "\n\n"
    "@app.get('/models')\n"
    "async def get_models_info() -> Dict[str, Any]:\n"
    '    """Return metrics for both trained models read from reports/*.json.\n'
    "\n"
    "    Numbers are never hardcoded -- always read from files at request time.\n"
    "    Missing files are reported as null rather than raising an error.\n"
    '    """\n'
    "    reports_dir = MODELS_DIR.parent / 'reports'\n"
    "\n"
    "    def _load_json(path):\n"
    "        try:\n"
    "            return json.loads(path.read_text(encoding='utf-8'))\n"
    "        except Exception:\n"
    "            return None\n"
    "\n"
    "    xgb_metrics = _load_json(reports_dir / 'xgb_metrics.json')\n"
    "    rf_metrics   = _load_json(reports_dir / 'rf_metrics.json')\n"
    "\n"
    "    try:\n"
    "        from backend.config import MODEL_BACKEND\n"
    "        active_backend = MODEL_BACKEND\n"
    "    except Exception:\n"
    "        active_backend = 'random_forest'\n"
    "\n"
    "    return {\n"
    "        'active_model':  active_backend,\n"
    "        'xgboost':       xgb_metrics,\n"
    "        'random_forest': rf_metrics,\n"
    "    }\n"
    "\n"
)

marker = "@app.get(\"/stats\")"
if marker in content:
    content = content.replace(marker, new_endpoint + marker, 1)
    open('backend/main.py', 'w', encoding='utf-8').write(content)
    print("OK: /models endpoint injected")
else:
    print("MARKER NOT FOUND:", repr(marker[:50]))
    sys.exit(1)

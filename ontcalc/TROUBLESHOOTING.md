# OntoCalc Troubleshooting Guide

## Common Issues and Solutions

### 1. "No module named 'anthropic'" Error

**Symptoms:**
- AI Logs tab shows: `Error: No module named 'anthropic'`
- AI mapping fails and falls back to string similarity

**Solutions:**

#### Option A: Install anthropic module
```bash
cd ontcalc
pip install "anthropic>=0.40.0"
```

#### Option B: Use the setup script
```bash
cd ontcalc
./setup.sh
```

#### Option C: Use a virtual environment (recommended)
```bash
cd ontcalc
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**After installing, you MUST restart the web server:**
```bash
# Stop the current server (Ctrl+C)
# Then restart:
python3 web_editor.py
```

### 2. Web Server Not Refreshing After Code Changes

**Symptoms:**
- Modified code but behavior hasn't changed
- Old errors still appearing after fixes

**Solution:**
Always restart the Flask server after making changes:
```bash
# 1. Stop the server: Press Ctrl+C in the terminal
# 2. Restart the server:
python3 web_editor.py
```

**Note:** Even with `debug=True`, Python import caching can cause issues. A full restart is recommended.

### 3. AI API Key Not Working

**Symptoms:**
- AI Logs show "No API key provided"
- Even after setting ANTHROPIC_API_KEY

**Solutions:**

#### Check if environment variable is set:
```bash
echo $ANTHROPIC_API_KEY
```

#### Set it in the current session:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

#### Or set it permanently:
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

Then reload:
```bash
source ~/.bashrc  # or ~/.zshrc
```

#### Alternative: Pass API key in web UI
Enter your API key in the "Anthropic API Key" field in the web editor.

### 4. Anthropic Version Compatibility Error

**Symptoms:**
- Error: `Client.__init__() got an unexpected keyword argument 'proxies'`
- AI mapping fails even with anthropic installed

**Cause:**
Old version of anthropic library (0.39.0 or earlier) is incompatible with the current API.

**Solution:**
Upgrade to the latest version:
```bash
pip install --upgrade "anthropic>=0.40.0"

# Then restart the web server
python3 web_editor.py
```

**Verify the version:**
```bash
python3 -c "import anthropic; print(anthropic.__version__)"
# Should show 0.40.0 or higher
```

### 5. Module Import Cache Issues

**Symptoms:**
- Changes to Python files not taking effect
- `anthropic` installed but still shows import error

**Solution:**
Clear Python cache and restart:
```bash
# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Restart server
python3 web_editor.py
```

### 6. Port 5000 Already in Use

**Symptoms:**
- Error: `Address already in use`
- Web server won't start

**Solutions:**

#### Option A: Kill existing process
```bash
# Find process using port 5000
lsof -i :5000

# Kill it (replace PID with actual process ID)
kill -9 PID
```

#### Option B: Use a different port
Edit `web_editor.py` line 156:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Changed from 5000 to 5001
```

### 7. Permission Denied Errors

**Symptoms:**
- Can't execute `./setup.sh`
- Permission denied when running scripts

**Solution:**
```bash
chmod +x setup.sh
./setup.sh
```

### 8. Compute Path Errors

**Symptoms:**
- `Node 'X' not found in graph 'Y'`
- Available nodes list shows different IDs

**Common Causes:**
- Node IDs change after merge operations
- Using original IDs instead of merged IDs

**Solution:**
Check the graph structure first:
```ontocalc
# For merged graphs, use the merged node IDs
# Check "Graphs" tab in web UI to see actual node IDs
```

Or compute before merging:
```ontocalc
# Method 1: Compute before merge
compute EmissionData.Entry_1.emissions using formula "activity * emissionFactor";

# Method 2: Then merge
merge FactoryData, EmissionData as M using policy merge-policy;
```

### 9. Missing Dependencies

**Symptoms:**
- Various import errors
- Module not found errors

**Solution:**
Reinstall all dependencies:
```bash
cd ontcalc
pip install -r requirements.txt --force-reinstall
```

### 10. Getting Help

If you're still experiencing issues:

1. **Check the AI Logs tab** - It provides detailed error messages
2. **Check terminal output** - Flask prints useful debug information
3. **Review the examples** - `examples/toy_example.onto` and `examples/ghg_ontocalc.onto`
4. **Read the documentation** - `README.md` and `spec/ontocalc_bnf.txt`

### 11. Quick Diagnosis Script

Run this to check your environment:
```bash
cd ontcalc
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")
print(f"Python Path: {sys.executable}")

try:
    import flask
    print(f"✓ Flask: {flask.__version__}")
except ImportError:
    print("✗ Flask: NOT INSTALLED")

try:
    import anthropic
    print(f"✓ Anthropic: {anthropic.__version__}")
    if hasattr(anthropic, '__version__') and anthropic.__version__ < '0.40.0':
        print(f"⚠ Warning: anthropic {anthropic.__version__} is outdated. Upgrade with: pip install --upgrade 'anthropic>=0.40.0'")
except ImportError:
    print("✗ Anthropic: NOT INSTALLED")

import os
api_key = os.environ.get('ANTHROPIC_API_KEY')
if api_key:
    print(f"✓ API Key: Set (length: {len(api_key)})")
else:
    print("○ API Key: Not set (will use fallback)")
EOF
```

This will show you what's installed and configured.

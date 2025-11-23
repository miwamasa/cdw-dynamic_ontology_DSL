#!/bin/bash
# OntoCalc Setup Script

echo "🔮 Setting up OntoCalc..."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1)
echo "✓ Python: $python_version"

# Create virtual environment (optional but recommended)
read -p "Create virtual environment? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
    echo ""
    echo "To activate it, run:"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Activate now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]
    then
        source venv/bin/activate
    fi
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the examples:"
echo "  python3 ontocalc_full.py examples/toy_example.onto"
echo "  python3 ontocalc_full.py examples/ghg_ontocalc.onto"
echo ""
echo "To start the web editor:"
echo "  python3 web_editor.py"
echo "  Then open http://localhost:5000 in your browser"
echo ""
echo "For AI-assisted mapping, set your API key:"
echo "  export ANTHROPIC_API_KEY='your-api-key-here'"
echo ""

"""OntoCalc Web Editor and Execution Environment

A Flask-based web application for editing and executing OntoCalc DSL scripts.
"""

from flask import Flask, render_template, request, jsonify
import json
import traceback
from ontocalc_full import Lexer, Parser, Executor, AIMapper
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('editor.html')

@app.route('/api/execute', methods=['POST'])
def execute_script():
    """Execute OntoCalc script and return results"""
    try:
        data = request.get_json()
        script = data.get('script', '')
        api_key = data.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')

        # Lex
        lexer = Lexer(script)
        tokens = lexer.tokenize()

        # Parse
        parser = Parser(tokens)
        ast = parser.parse()

        # Execute
        ai_mapper = AIMapper(api_key=api_key)
        executor = Executor(ai_mapper=ai_mapper)
        executor.execute_program(ast)

        # Collect results
        results = {
            'success': True,
            'graphs': {name: executor.export_graph_json(name) for name in executor.graphs.keys()},
            'schemas': {name: {'name': name, 'decls': len(schema.decls)}
                       for name, schema in executor.schemas.items()},
            'mappings': {
                'suggested': len(executor.mappings),
                'approved': len(executor.approved),
                'list': executor.approved
            },
            'provenance': executor.provenance_log,
            'ai_logs': ai_mapper.ai_logs,  # Include AI interaction logs
            'tokens_count': len(tokens) - 1  # Exclude EOF
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 400

@app.route('/api/validate', methods=['POST'])
def validate_script():
    """Validate OntoCalc script syntax"""
    try:
        data = request.get_json()
        script = data.get('script', '')

        # Try to lex and parse
        lexer = Lexer(script)
        tokens = lexer.tokenize()

        parser = Parser(tokens)
        ast = parser.parse()

        return jsonify({
            'success': True,
            'valid': True,
            'tokens_count': len(tokens) - 1,
            'statements_count': len(ast)
        })

    except Exception as e:
        return jsonify({
            'success': True,
            'valid': False,
            'error': str(e),
            'error_type': type(e).__name__
        })

@app.route('/api/examples')
def get_examples():
    """Return example scripts"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(script_dir, 'examples')

    examples = {
        'toy_example': {
            'name': 'Toy Example: Animals and Pets',
            'description': 'Simple ontology transformation from animal classification to pet registry',
            'script': '''# Toy Example: Animals to Pets transformation
schema AnimalOntology {
  class Animal;
  class Mammal;
  prop species: Animal -> String;
  prop age: Animal -> Decimal;
  prop habitat: Animal -> String;
}

schema PetOntology {
  class Pet;
  class DomesticAnimal;
  prop name: Pet -> String;
  prop age: Pet -> Decimal;
  prop owner: Pet -> String;
}

instance AnimalData {
  Mammal Dog1 {
    species = "Canis familiaris";
    age = 3;
    habitat = "domestic";
  }
  Mammal Cat1 {
    species = "Felis catus";
    age = 2;
    habitat = "domestic";
  }
}

# Transfer from Animal ontology to Pet ontology
map AnimalOntology.Mammal -> PetOntology.Pet (confidence=0.9);
map AnimalOntology.age -> PetOntology.age (confidence=1.0);

approve-mapping all where confidence >= 0.85;

transfer AnimalData from AnimalOntology to PetOntology as PetData;
'''
        }
    }

    # Try to load GHG example from file
    ghg_file = os.path.join(examples_dir, 'ghg_ontocalc.onto')
    try:
        if os.path.exists(ghg_file):
            with open(ghg_file, 'r') as f:
                examples['ghg_example'] = {
                    'name': 'GHG Example: Production to Emissions',
                    'description': 'Factory production data to GHG emission reports',
                    'script': f.read()
                }
    except Exception as e:
        print(f"Warning: Could not load GHG example: {e}")

    return jsonify(examples)

if __name__ == '__main__':
    print("=" * 60)
    print("🔮 OntoCalc Web Editor Starting...")
    print("=" * 60)

    # Check dependencies
    import sys
    print(f"Python: {sys.version}")
    print(f"Python Path: {sys.executable}")

    try:
        import flask
        print(f"✓ Flask: {flask.__version__}")
    except ImportError:
        print("✗ Flask: NOT INSTALLED")
        print("  Run: pip install Flask==3.0.0")
        sys.exit(1)

    try:
        import anthropic
        print(f"✓ Anthropic: {anthropic.__version__}")
    except ImportError:
        print("⚠ Anthropic: NOT INSTALLED")
        print("  AI mapping will use string similarity fallback")
        print("  To enable AI mapping, run: pip install anthropic==0.39.0")

    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if api_key:
        print(f"✓ API Key: Configured (length: {len(api_key)})")
    else:
        print("○ API Key: Not set (using fallback mode)")
        print("  To enable AI mapping, set: export ANTHROPIC_API_KEY='your-key'")

    print("=" * 60)
    print("Starting server on http://0.0.0.0:5000")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    app.run(debug=True, host='0.0.0.0', port=5000)

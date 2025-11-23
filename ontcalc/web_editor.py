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
        },
        'ghg_example': {
            'name': 'GHG Example: Production to Emissions',
            'description': 'Factory production data to GHG emission reports',
            'script': open('/home/user/cdw-dynamic_ontology_DSL/ontcalc/examples/ghg_ontocalc.onto', 'r').read()
        }
    }

    return jsonify(examples)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

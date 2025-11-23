# OntoCalc - Dynamic Ontology DSL

OntoCalc is a Domain-Specific Language (DSL) for dynamic ontology operations with a focus on practical, reproducible, and auditable ontology transformations.

## 🎯 Overview

OntoCalc enables:
- **Declarative ontology definitions** using schemas and instances
- **Category-theoretic operations** (merge, compose, transfer) with provenance tracking
- **AI-assisted mapping** suggestions using Claude API
- **Safe arithmetic computation** with decimal precision
- **Web-based editor** for interactive development

## 📁 Project Structure

```
cdw-dynamic_ontology_DSL/
├── README.md                          # This file
├── instructions.md                    # Project requirements
├── related_documents/                 # Background information
│   └── the_ontocalc_summary.md       # Theoretical foundations
└── ontcalc/                          # Main implementation
    ├── README.md                      # Package documentation
    ├── requirements.txt               # Python dependencies
    ├── spec/
    │   ├── ontocalc_bnf.txt          # DSL grammar (BNF)
    │   └── ontocalc_ref.txt          # Reference manual
    ├── prototype/
    │   └── ontocalc_executor.py      # Simple prototype
    ├── ontocalc_full.py              # Complete interpreter
    ├── web_editor.py                 # Flask web application
    ├── templates/
    │   └── editor.html               # Web editor UI
    ├── examples/
    │   ├── toy_example.onto          # Example 1: Animals→Pets
    │   └── ghg_ontocalc.onto         # Example 2: Production→GHG
    └── sparql/
        └── sparql_checks.sparql      # Validation queries
```

## 🚀 Quick Start

### Installation

#### Option 1: Automated Setup (Recommended)

```bash
cd ontcalc
./setup.sh
```

The setup script will:
- Check your Python version
- Optionally create a virtual environment
- Install all dependencies (Flask, anthropic)
- Display usage instructions

#### Option 2: Manual Installation

```bash
cd ontcalc
pip install -r requirements.txt
```

### AI Configuration (Optional)

For AI-assisted ontology mapping using Claude Haiku 4.5:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Note**: The system works without an API key using string similarity fallback. With an API key, it uses **Claude Haiku 4.5** for intelligent mapping suggestions with rationale.

### Running Examples

#### Command Line

Execute OntoCalc scripts from the command line:

```bash
# Run the toy example
python3 ontocalc_full.py examples/toy_example.onto

# Run the GHG example
python3 ontocalc_full.py examples/ghg_ontocalc.onto
```

#### Web Editor

Launch the interactive web editor:

```bash
python3 web_editor.py
```

Then open your browser to `http://localhost:5000`

The web editor provides:
- **Syntax highlighting** for OntoCalc scripts
- **Real-time validation** of DSL syntax
- **Interactive execution** with visual results
- **Example scripts** to get started quickly
- **Provenance tracking** for all operations

## 📖 DSL Features

### Core Operations

1. **Schema Definition**
```ontocalc
schema AnimalOntology {
  class Animal;
  class Mammal;
  prop species: Animal -> String;
  prop age: Animal -> Decimal;
}
```

2. **Instance Creation**
```ontocalc
instance AnimalData {
  Mammal Dog1 {
    species = "Canis familiaris";
    age = 3;
  }
}
```

3. **Mapping**
```ontocalc
map AnimalOntology.Mammal -> PetOntology.Pet (confidence=0.95);
approve-mapping all where confidence >= 0.85;
```

4. **Transfer (NEW!)** - Transform data between ontologies
```ontocalc
transfer AnimalData from AnimalOntology to PetOntology as PetData;
```

5. **Merge** - Combine graphs using category-theoretic pushout
```ontocalc
merge FactoryData, EmissionData as M using policy merge-policy;
```

6. **Compute** - Safe arithmetic evaluation
```ontocalc
compute EmissionData.Entry_1.emissions using formula "activity * emissionFactor";
```

### Key Features

- **Type Safety**: Static type checking for schemas
- **Provenance**: Every operation is logged with timestamp and rationale
- **Decimal Precision**: All numeric computations use `Decimal` for accuracy
- **AI Integration**: Optional AI-assisted mapping suggestions
- **Extensible**: Easy to add new operations and validators

## 📚 Examples

### Example 1: Toy Example - Animals to Pets

Demonstrates basic ontology transformation:

```ontocalc
# Define source ontology
schema AnimalOntology {
  class Mammal;
  prop species: Animal -> String;
  prop age: Animal -> Decimal;
}

# Define target ontology
schema PetOntology {
  class Pet;
  prop petName: Pet -> String;
  prop age: Pet -> Decimal;
}

# Create instances
instance AnimalData {
  Mammal Dog1 {
    species = "Canis familiaris";
    age = 3;
    weight = 25.5;
  }
}

# Map and transfer
map AnimalOntology.Mammal -> PetOntology.Pet (confidence=0.95);
approve-mapping all where confidence >= 0.85;
transfer AnimalData from AnimalOntology to PetOntology as PetData;
```

**Purpose**: Intuitive demonstration of ontology operations suitable for learning.

### Example 2: GHG Reporting

Real-world scenario transforming production data to GHG emission reports:

```ontocalc
# Factory production tracking
schema FactoryModule {
  class ProductionBatch;
  prop quantity: ProductionBatch -> Decimal(unit="count");
}

# GHG emission reporting
schema EmissionModule {
  class EmissionEntry;
  prop activity: EmissionEntry -> Decimal(unit="count");
  prop emissionFactor: EmissionEntry -> Decimal(unit="kgCO2e_per_unit");
  prop emissions: EmissionEntry -> Decimal(unit="kgCO2e");
}

# Map production to emissions
map FactoryModule.ProductionBatch -> EmissionModule.EmissionEntry (confidence=0.92);
map FactoryModule.quantity -> EmissionModule.activity (confidence=0.98);

# Compute emissions
compute EmissionData.Entry_1.emissions using formula "activity * emissionFactor";
# Result: 1000 * 0.75 = 750.00 kgCO2e
```

**Purpose**: Demonstrates practical business use case with verifiable calculations.

## 🏗️ Architecture

### Lexer → Parser → AST → Executor

1. **Lexer** (`Lexer` class): Tokenizes input text
2. **Parser** (`Parser` class): Builds Abstract Syntax Tree (AST)
3. **Executor** (`Executor` class): Executes AST nodes with:
   - Graph model for in-memory representation
   - Union-Find for merge operations (coequalizer)
   - Provenance logging for auditability
   - Safe formula evaluation (sandboxed)

### Category-Theoretic Operations

- **Merge**: Implements pushout/coequalizer using Union-Find
- **Transfer**: Maps instances from source to target ontology
- **Compose**: Connects ontologies via shared interfaces

### AI Integration

The `AIMapper` class provides:
- **Intelligent mapping suggestions** using Claude Haiku 4.5 API
- **Detailed logging** of all AI interactions (prompts, responses, tokens)
- **Automatic fallback** to string similarity when API unavailable
- **Confidence scoring** for all suggestions with rationale
- **Human-in-the-loop** approval workflow
- **Token usage tracking** for cost monitoring

AI logs are fully visible in the web editor's "AI Logs" tab, showing:
- Complete prompts and responses
- Token usage statistics
- Parsed mapping results with confidence scores
- Error messages and fallback behavior

## 🔧 Technical Details

### Safe Formula Evaluation

OntoCalc uses a restricted `eval()` with:
- Only arithmetic operations allowed
- No code injection possible (no semicolons, imports, etc.)
- `Decimal` type for precision
- Limited function whitelist: `pow`, `min`, `max`, `abs`

### Provenance Tracking

Every operation records:
- Operation type
- Input parameters
- Output results
- Timestamp (UTC)
- Source graph/node references

Example provenance entry:
```json
{
  "op": "compute",
  "graph": "EmissionData",
  "node": "Entry_1",
  "prop": "emissions",
  "value": "750.00",
  "formula": "activity * emissionFactor",
  "when": "2025-11-23T10:30:00.000000"
}
```

## 🧪 Testing

Both examples include comprehensive test coverage:

1. **Syntax Validation**: Parser correctly handles all DSL constructs
2. **Semantic Validation**: Type checking and consistency
3. **Computation Accuracy**: Decimal arithmetic verification
4. **Provenance Completeness**: All operations logged

Run tests:
```bash
# Test toy example
python3 ontocalc_full.py examples/toy_example.onto

# Test GHG example
python3 ontocalc_full.py examples/ghg_ontocalc.onto

# Both should complete without errors
```

## 🤝 Contributing

### Adding New Operations

To add a new operation:

1. Add to BNF grammar (`spec/ontocalc_bnf.txt`)
2. Add token type to `TokenType` enum
3. Add AST node class
4. Implement parser method
5. Implement executor method
6. Add example usage

### AI Integration

The system supports AI-assisted operations via:
- Anthropic Claude API for mapping suggestions
- Extensible `AIMapper` interface
- Graceful degradation to rule-based fallback

## 📄 License

This project is part of the CDW Dynamic Ontology DSL research initiative.

## 🙏 Acknowledgments

Based on category-theoretic foundations described in `related_documents/the_ontocalc_summary.md`, implementing:
- Graph algebra operations
- Pushout/pullback constructions
- Provenance tracking
- Reproducible transformations

## 🔧 Troubleshooting

Experiencing issues? Check the detailed troubleshooting guide:

📖 **[TROUBLESHOOTING.md](ontcalc/TROUBLESHOOTING.md)**

Common issues covered:
- "No module named 'anthropic'" errors
- Web server not refreshing after code changes
- API key configuration
- Module import cache issues
- Port conflicts
- And more...

Quick diagnosis:
```bash
cd ontcalc
python3 web_editor.py
# Check the startup messages for any warnings
```

## 📞 Support

For issues or questions:
1. **Check [TROUBLESHOOTING.md](ontcalc/TROUBLESHOOTING.md)** for common issues
2. Review the examples in `ontcalc/examples/`
3. Read the BNF grammar in `spec/ontocalc_bnf.txt`
4. Check the theoretical foundations in `related_documents/`

---

**Version**: 1.0.0
**Last Updated**: 2025-11-23

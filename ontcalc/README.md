OntoCalc Package - Prototype
============================

Contents:
- spec/ontocalc_bnf.txt          : DSL BNF excerpt
- spec/ontocalc_ref.txt         : Reference manual (abridged)
- examples/ghg_ontocalc.onto    : Example OntoCalc script for the GHG case
- prototype/ontocalc_executor.py: Lightweight Python prototype executor (parser+merge+compute)
- sparql/sparql_checks.sparql   : SPARQL checks for numeric consistency
- README.md                     : this file

How to run prototype demo (no external deps):
- Run python3 prototype/ontocalc_executor.py
  It will execute the embedded demo, print approved mappings, merged nodes, computed emissions,
  and export merged graph as JSON to stdout.

Notes:
- This is a prototype for demonstration and testing. It is NOT a production-ready parser or reasoner.
- For production, integrate with RDFLib / Apache Jena / OWL reasoners and secure evaluation for formulas.

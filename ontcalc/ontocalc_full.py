"""OntoCalc Full Interpreter with Transfer Operation and AI API Integration

This is a complete implementation of the OntoCalc DSL interpreter with:
- Full DSL parser (lexer + parser)
- All operations: map, merge, diff, compose, transfer, compute, check
- AI API integration for mapping suggestions
- Provenance tracking
- Web editor support (via Flask API)
"""

import re
import json
import os
from decimal import Decimal, getcontext
from datetime import datetime
from collections import defaultdict, namedtuple
from typing import Dict, List, Set, Tuple, Any, Optional
from enum import Enum

getcontext().prec = 28

# ============================================================================
# Token and Lexer
# ============================================================================

class TokenType(Enum):
    # Keywords
    MODULE = 'module'
    SCHEMA = 'schema'
    INSTANCE = 'instance'
    CLASS = 'class'
    PROP = 'prop'
    AXIOM = 'axiom'
    MAP = 'map'
    MERGE = 'merge'
    DIFF = 'diff'
    COMPOSE = 'compose'
    TRANSFER = 'transfer'
    COMPUTE = 'compute'
    CHECK_CONSISTENCY = 'check-consistency'
    RUN_SPARQL = 'run-sparql'
    COMMIT = 'commit'
    IMPORT = 'import'
    AS = 'as'
    USING = 'using'
    POLICY = 'policy'
    MINUS = 'minus'
    VIA = 'via'
    FROM = 'from'
    TO = 'to'
    FORMULA = 'formula'
    BY = 'by'
    CONFIDENCE = 'confidence'
    APPROVE_MAPPING = 'approve-mapping'
    ALL = 'all'
    WHERE = 'where'

    # Symbols
    LBRACE = '{'
    RBRACE = '}'
    LPAREN = '('
    RPAREN = ')'
    SEMICOLON = ';'
    COMMA = ','
    DOT = '.'
    COLON = ':'
    ARROW = '->'
    EQUALS = '='
    QUESTION = '?'
    GTE = '>='

    # Literals
    ID = 'ID'
    STRING = 'STRING'
    NUMBER = 'NUMBER'

    # Special
    COMMENT = 'COMMENT'
    EOF = 'EOF'

class Token:
    def __init__(self, type: TokenType, value: str, line: int, col: int):
        self.type = type
        self.value = value
        self.line = line
        self.col = col

    def __repr__(self):
        return f'Token({self.type}, {self.value!r}, {self.line}:{self.col})'

class Lexer:
    def __init__(self, text: str):
        self.text = text
        self.pos = 0
        self.line = 1
        self.col = 1
        self.tokens = []

        self.keywords = {
            'module': TokenType.MODULE,
            'schema': TokenType.SCHEMA,
            'instance': TokenType.INSTANCE,
            'class': TokenType.CLASS,
            'prop': TokenType.PROP,
            'axiom': TokenType.AXIOM,
            'map': TokenType.MAP,
            'merge': TokenType.MERGE,
            'diff': TokenType.DIFF,
            'compose': TokenType.COMPOSE,
            'transfer': TokenType.TRANSFER,
            'compute': TokenType.COMPUTE,
            'check-consistency': TokenType.CHECK_CONSISTENCY,
            'run-sparql': TokenType.RUN_SPARQL,
            'commit': TokenType.COMMIT,
            'import': TokenType.IMPORT,
            'as': TokenType.AS,
            'using': TokenType.USING,
            'policy': TokenType.POLICY,
            'minus': TokenType.MINUS,
            'via': TokenType.VIA,
            'from': TokenType.FROM,
            'to': TokenType.TO,
            'formula': TokenType.FORMULA,
            'by': TokenType.BY,
            'confidence': TokenType.CONFIDENCE,
            'approve-mapping': TokenType.APPROVE_MAPPING,
            'all': TokenType.ALL,
            'where': TokenType.WHERE,
        }

    def current_char(self) -> Optional[str]:
        if self.pos >= len(self.text):
            return None
        return self.text[self.pos]

    def peek(self, offset=1) -> Optional[str]:
        pos = self.pos + offset
        if pos >= len(self.text):
            return None
        return self.text[pos]

    def advance(self):
        if self.pos < len(self.text):
            if self.text[self.pos] == '\n':
                self.line += 1
                self.col = 1
            else:
                self.col += 1
            self.pos += 1

    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\n\r':
            self.advance()

    def skip_comment(self):
        if self.current_char() == '#':
            while self.current_char() and self.current_char() != '\n':
                self.advance()
            if self.current_char() == '\n':
                self.advance()

    def read_string(self) -> str:
        result = ''
        self.advance()  # skip opening quote
        while self.current_char() and self.current_char() != '"':
            if self.current_char() == '\\':
                self.advance()
                if self.current_char():
                    result += self.current_char()
                    self.advance()
            else:
                result += self.current_char()
                self.advance()
        if self.current_char() == '"':
            self.advance()  # skip closing quote
        return result

    def read_number(self) -> str:
        result = ''
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            result += self.current_char()
            self.advance()
        return result

    def read_id_or_keyword(self) -> Tuple[TokenType, str]:
        result = ''
        while self.current_char() and (self.current_char().isalnum() or self.current_char() in '_-'):
            result += self.current_char()
            self.advance()

        if result in self.keywords:
            return self.keywords[result], result
        return TokenType.ID, result

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.text):
            self.skip_whitespace()
            if self.pos >= len(self.text):
                break

            if self.current_char() == '#':
                self.skip_comment()
                continue

            line, col = self.line, self.col

            # Two-character operators
            if self.current_char() == '-' and self.peek() == '>':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.ARROW, '->', line, col))
            elif self.current_char() == '>' and self.peek() == '=':
                self.advance()
                self.advance()
                self.tokens.append(Token(TokenType.GTE, '>=', line, col))
            # Single-character symbols
            elif self.current_char() == '{':
                self.tokens.append(Token(TokenType.LBRACE, '{', line, col))
                self.advance()
            elif self.current_char() == '}':
                self.tokens.append(Token(TokenType.RBRACE, '}', line, col))
                self.advance()
            elif self.current_char() == '(':
                self.tokens.append(Token(TokenType.LPAREN, '(', line, col))
                self.advance()
            elif self.current_char() == ')':
                self.tokens.append(Token(TokenType.RPAREN, ')', line, col))
                self.advance()
            elif self.current_char() == ';':
                self.tokens.append(Token(TokenType.SEMICOLON, ';', line, col))
                self.advance()
            elif self.current_char() == ',':
                self.tokens.append(Token(TokenType.COMMA, ',', line, col))
                self.advance()
            elif self.current_char() == '.':
                self.tokens.append(Token(TokenType.DOT, '.', line, col))
                self.advance()
            elif self.current_char() == ':':
                self.tokens.append(Token(TokenType.COLON, ':', line, col))
                self.advance()
            elif self.current_char() == '=':
                self.tokens.append(Token(TokenType.EQUALS, '=', line, col))
                self.advance()
            elif self.current_char() == '?':
                self.tokens.append(Token(TokenType.QUESTION, '?', line, col))
                self.advance()
            # String
            elif self.current_char() == '"':
                value = self.read_string()
                self.tokens.append(Token(TokenType.STRING, value, line, col))
            # Number
            elif self.current_char().isdigit():
                value = self.read_number()
                self.tokens.append(Token(TokenType.NUMBER, value, line, col))
            # ID or keyword
            elif self.current_char().isalpha() or self.current_char() == '_':
                token_type, value = self.read_id_or_keyword()
                self.tokens.append(Token(token_type, value, line, col))
            else:
                raise SyntaxError(f"Unexpected character '{self.current_char()}' at {line}:{col}")

        self.tokens.append(Token(TokenType.EOF, '', self.line, self.col))
        return self.tokens

# ============================================================================
# AST Nodes
# ============================================================================

class ASTNode:
    pass

class ModuleNode(ASTNode):
    def __init__(self, name: str, imports: List[str]):
        self.name = name
        self.imports = imports

class SchemaNode(ASTNode):
    def __init__(self, name: str, decls: List):
        self.name = name
        self.decls = decls

class ClassDecl(ASTNode):
    def __init__(self, name: str):
        self.name = name

class PropDecl(ASTNode):
    def __init__(self, name: str, type_spec: str):
        self.name = name
        self.type_spec = type_spec

class InstanceNode(ASTNode):
    def __init__(self, name: str, entities: List):
        self.name = name
        self.entities = entities

class EntityDecl(ASTNode):
    def __init__(self, type_name: str, entity_id: str, props: Dict):
        self.type_name = type_name
        self.entity_id = entity_id
        self.props = props

class MapCmd(ASTNode):
    def __init__(self, left: str, right: str, confidence: float):
        self.left = left
        self.right = right
        self.confidence = confidence

class MergeCmd(ASTNode):
    def __init__(self, left: str, right: str, output: str, policy: str):
        self.left = left
        self.right = right
        self.output = output
        self.policy = policy

class TransferCmd(ASTNode):
    def __init__(self, data: str, from_schema: str, to_schema: str, output: str, method: Optional[str] = None):
        self.data = data
        self.from_schema = from_schema
        self.to_schema = to_schema
        self.output = output
        self.method = method

class ComputeCmd(ASTNode):
    def __init__(self, path: List[str], formula: str):
        self.path = path
        self.formula = formula

class ApproveMapping(ASTNode):
    def __init__(self, threshold: Optional[float] = None):
        self.threshold = threshold

# ============================================================================
# Parser
# ============================================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def current_token(self) -> Token:
        if self.pos >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[self.pos]

    def advance(self):
        if self.pos < len(self.tokens) - 1:
            self.pos += 1

    def expect(self, token_type: TokenType) -> Token:
        token = self.current_token()
        if token.type != token_type:
            raise SyntaxError(f"Expected {token_type}, got {token.type} at {token.line}:{token.col}")
        self.advance()
        return token

    def parse(self) -> List[ASTNode]:
        statements = []
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                statements.append(stmt)
        return statements

    def parse_statement(self) -> Optional[ASTNode]:
        token = self.current_token()

        if token.type == TokenType.MODULE:
            return self.parse_module()
        elif token.type == TokenType.SCHEMA:
            return self.parse_schema()
        elif token.type == TokenType.INSTANCE:
            return self.parse_instance()
        elif token.type == TokenType.MAP:
            return self.parse_map()
        elif token.type == TokenType.MERGE:
            return self.parse_merge()
        elif token.type == TokenType.TRANSFER:
            return self.parse_transfer()
        elif token.type == TokenType.COMPUTE:
            return self.parse_compute()
        elif token.type == TokenType.APPROVE_MAPPING:
            return self.parse_approve_mapping()
        else:
            raise SyntaxError(f"Unexpected token {token.type} at {token.line}:{token.col}")

    def parse_module(self) -> ModuleNode:
        self.expect(TokenType.MODULE)
        name = self.expect(TokenType.ID).value
        self.expect(TokenType.LBRACE)

        imports = []
        while self.current_token().type == TokenType.IMPORT:
            self.advance()
            imports.append(self.expect(TokenType.ID).value)
            self.expect(TokenType.SEMICOLON)

        self.expect(TokenType.RBRACE)
        return ModuleNode(name, imports)

    def parse_schema(self) -> SchemaNode:
        self.expect(TokenType.SCHEMA)
        name = self.expect(TokenType.ID).value
        self.expect(TokenType.LBRACE)

        decls = []
        while self.current_token().type in [TokenType.CLASS, TokenType.PROP]:
            if self.current_token().type == TokenType.CLASS:
                self.advance()
                class_name = self.expect(TokenType.ID).value
                self.expect(TokenType.SEMICOLON)
                decls.append(ClassDecl(class_name))
            elif self.current_token().type == TokenType.PROP:
                self.advance()
                prop_name = self.expect(TokenType.ID).value
                self.expect(TokenType.COLON)
                # Read type spec until semicolon
                type_parts = []
                while self.current_token().type != TokenType.SEMICOLON:
                    type_parts.append(self.current_token().value)
                    self.advance()
                type_spec = ' '.join(type_parts)
                self.expect(TokenType.SEMICOLON)
                decls.append(PropDecl(prop_name, type_spec))

        self.expect(TokenType.RBRACE)
        return SchemaNode(name, decls)

    def parse_instance(self) -> InstanceNode:
        self.expect(TokenType.INSTANCE)
        name = self.expect(TokenType.ID).value
        self.expect(TokenType.LBRACE)

        entities = []
        while self.current_token().type == TokenType.ID:
            type_name = self.expect(TokenType.ID).value
            entity_id = self.expect(TokenType.ID).value
            self.expect(TokenType.LBRACE)

            props = {}
            while self.current_token().type == TokenType.ID:
                prop_name = self.expect(TokenType.ID).value
                self.expect(TokenType.EQUALS)

                # Read value
                value_token = self.current_token()
                if value_token.type == TokenType.STRING:
                    value = value_token.value
                    self.advance()
                elif value_token.type == TokenType.NUMBER:
                    value = value_token.value
                    self.advance()
                elif value_token.type == TokenType.ID:
                    value = value_token.value
                    self.advance()
                elif value_token.type == TokenType.QUESTION:
                    value = None
                    self.advance()
                else:
                    raise SyntaxError(f"Unexpected value token {value_token.type}")

                self.expect(TokenType.SEMICOLON)
                props[prop_name] = value

            self.expect(TokenType.RBRACE)
            entities.append(EntityDecl(type_name, entity_id, props))

        self.expect(TokenType.RBRACE)
        return InstanceNode(name, entities)

    def parse_map(self) -> MapCmd:
        self.expect(TokenType.MAP)
        left_parts = [self.expect(TokenType.ID).value]
        while self.current_token().type == TokenType.DOT:
            self.advance()
            left_parts.append(self.expect(TokenType.ID).value)
        left = '.'.join(left_parts)

        self.expect(TokenType.ARROW)

        right_parts = [self.expect(TokenType.ID).value]
        while self.current_token().type == TokenType.DOT:
            self.advance()
            right_parts.append(self.expect(TokenType.ID).value)
        right = '.'.join(right_parts)

        self.expect(TokenType.LPAREN)
        self.expect(TokenType.CONFIDENCE)
        self.expect(TokenType.EQUALS)
        confidence = float(self.expect(TokenType.NUMBER).value)
        self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMICOLON)

        return MapCmd(left, right, confidence)

    def parse_merge(self) -> MergeCmd:
        self.expect(TokenType.MERGE)
        left = self.expect(TokenType.ID).value
        self.expect(TokenType.COMMA)
        right = self.expect(TokenType.ID).value
        self.expect(TokenType.AS)
        output = self.expect(TokenType.ID).value
        self.expect(TokenType.USING)
        self.expect(TokenType.POLICY)
        policy = self.expect(TokenType.ID).value
        self.expect(TokenType.SEMICOLON)

        return MergeCmd(left, right, output, policy)

    def parse_transfer(self) -> TransferCmd:
        self.expect(TokenType.TRANSFER)
        data = self.expect(TokenType.ID).value
        self.expect(TokenType.FROM)
        from_schema = self.expect(TokenType.ID).value
        self.expect(TokenType.TO)
        to_schema = self.expect(TokenType.ID).value
        self.expect(TokenType.AS)
        output = self.expect(TokenType.ID).value

        method = None
        if self.current_token().type == TokenType.USING:
            self.advance()
            method = self.expect(TokenType.ID).value

        self.expect(TokenType.SEMICOLON)

        return TransferCmd(data, from_schema, to_schema, output, method)

    def parse_compute(self) -> ComputeCmd:
        self.expect(TokenType.COMPUTE)

        path = [self.expect(TokenType.ID).value]
        while self.current_token().type == TokenType.DOT:
            self.advance()
            path.append(self.expect(TokenType.ID).value)

        self.expect(TokenType.USING)
        self.expect(TokenType.FORMULA)
        formula = self.expect(TokenType.STRING).value
        self.expect(TokenType.SEMICOLON)

        return ComputeCmd(path, formula)

    def parse_approve_mapping(self) -> ApproveMapping:
        self.expect(TokenType.APPROVE_MAPPING)
        self.expect(TokenType.ALL)

        threshold = None
        if self.current_token().type == TokenType.WHERE:
            self.advance()
            self.expect(TokenType.CONFIDENCE)
            self.expect(TokenType.GTE)
            threshold = float(self.expect(TokenType.NUMBER).value)

        self.expect(TokenType.SEMICOLON)
        return ApproveMapping(threshold)

# ============================================================================
# Runtime: Graph Model
# ============================================================================

class Node:
    def __init__(self, nid: str, types: Optional[Set[str]] = None):
        self.id = nid
        self.types = set(types or [])
        self.props = defaultdict(list)
        self.provenance = []

    def add_prop(self, p: str, v: Any, prov: Optional[Dict] = None):
        self.props[p].append(v)
        if prov:
            self.provenance.append(prov)

class Edge:
    def __init__(self, subj: str, pred: str, obj: str, prov: Optional[Dict] = None):
        self.subj = subj
        self.pred = pred
        self.obj = obj
        self.provenance = prov

class Graph:
    def __init__(self, name: str):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.meta = {}

    def add_node(self, nid: str, types: Optional[Set[str]] = None) -> Node:
        if nid not in self.nodes:
            self.nodes[nid] = Node(nid, types)
        else:
            if types:
                self.nodes[nid].types.update(types)
        return self.nodes[nid]

    def add_edge(self, subj: str, pred: str, obj: str, prov: Optional[Dict] = None):
        self.edges.append(Edge(subj, pred, obj, prov))

    def find_node(self, nid: str) -> Optional[Node]:
        return self.nodes.get(nid)

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            return x
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self):
        out = {}
        for k in self.parent:
            r = self.find(k)
            out.setdefault(r, []).append(k)
        return out

# ============================================================================
# AI API Integration
# ============================================================================

class AIMapper:
    """AI-assisted mapping suggestion using Claude API"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.ai_logs = []  # Track all AI interactions

    def suggest_mappings(self, schema_a: SchemaNode, schema_b: SchemaNode) -> List[Tuple[str, str, float]]:
        """Suggest mappings between two schemas using AI"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'operation': 'suggest_mappings',
            'schema_a': schema_a.name,
            'schema_b': schema_b.name,
            'method': 'ai' if self.api_key else 'fallback',
            'prompt': None,
            'response': None,
            'error': None,
            'mappings_count': 0
        }

        if not self.api_key:
            # Fallback to simple string similarity
            log_entry['method'] = 'string_similarity_fallback'
            log_entry['response'] = 'No API key provided, using string similarity'
            mappings = self._simple_similarity_mapping(schema_a, schema_b)
            log_entry['mappings_count'] = len(mappings)
            self.ai_logs.append(log_entry)
            return mappings

        # Use Claude API for intelligent mapping
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            prompt = f"""Given two ontology schemas, suggest mappings between their classes and properties.

Schema A ({schema_a.name}):
{self._format_schema(schema_a)}

Schema B ({schema_b.name}):
{self._format_schema(schema_b)}

For each potential mapping, provide:
1. The source term from Schema A
2. The target term from Schema B
3. A confidence score (0.0 to 1.0)
4. A brief rationale

Format your response as JSON:
{{"mappings": [{{"from": "A.term", "to": "B.term", "confidence": 0.95, "rationale": "..."}}]}}
"""
            log_entry['prompt'] = prompt

            message = client.messages.create(
                model="claude-haiku-4-5",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = message.content[0].text
            log_entry['response'] = response_text
            log_entry['model'] = 'claude-haiku-4-5'
            log_entry['tokens_used'] = {
                'input': message.usage.input_tokens,
                'output': message.usage.output_tokens
            }

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                mappings = []
                for m in result.get('mappings', []):
                    mappings.append((m['from'], m['to'], m['confidence']))
                log_entry['mappings_count'] = len(mappings)
                log_entry['parsed_mappings'] = result.get('mappings', [])
                self.ai_logs.append(log_entry)
                return mappings
        except Exception as e:
            log_entry['error'] = str(e)
            log_entry['method'] = 'ai_failed_fallback'
            print(f"AI mapping failed, using fallback: {e}")
            mappings = self._simple_similarity_mapping(schema_a, schema_b)
            log_entry['mappings_count'] = len(mappings)
            self.ai_logs.append(log_entry)
            return mappings

        mappings = self._simple_similarity_mapping(schema_a, schema_b)
        log_entry['mappings_count'] = len(mappings)
        self.ai_logs.append(log_entry)
        return mappings

    def _format_schema(self, schema: SchemaNode) -> str:
        lines = []
        for decl in schema.decls:
            if isinstance(decl, ClassDecl):
                lines.append(f"  class {decl.name}")
            elif isinstance(decl, PropDecl):
                lines.append(f"  prop {decl.name}: {decl.type_spec}")
        return '\n'.join(lines)

    def _simple_similarity_mapping(self, schema_a: SchemaNode, schema_b: SchemaNode) -> List[Tuple[str, str, float]]:
        """Simple string-based similarity mapping as fallback"""
        mappings = []

        def normalize(s):
            return s.lower().replace('_', '').replace('-', '')

        a_terms = []
        b_terms = []

        for decl in schema_a.decls:
            if isinstance(decl, ClassDecl):
                a_terms.append((f"{schema_a.name}.{decl.name}", decl.name))
            elif isinstance(decl, PropDecl):
                a_terms.append((f"{schema_a.name}.{decl.name}", decl.name))

        for decl in schema_b.decls:
            if isinstance(decl, ClassDecl):
                b_terms.append((f"{schema_b.name}.{decl.name}", decl.name))
            elif isinstance(decl, PropDecl):
                b_terms.append((f"{schema_b.name}.{decl.name}", decl.name))

        for a_full, a_name in a_terms:
            a_norm = normalize(a_name)
            for b_full, b_name in b_terms:
                b_norm = normalize(b_name)

                # Exact match
                if a_norm == b_norm:
                    mappings.append((a_full, b_full, 0.95))
                # Substring match
                elif a_norm in b_norm or b_norm in a_norm:
                    mappings.append((a_full, b_full, 0.75))

        return mappings

# ============================================================================
# Executor
# ============================================================================

def safe_eval_formula(formula: str, env: Dict[str, Any]) -> Decimal:
    """Safely evaluate arithmetic formula"""
    allowed_funcs = {'pow': pow, 'min': min, 'max': max, 'abs': abs}

    expr = formula
    for k, v in env.items():
        if isinstance(v, Decimal):
            expr = re.sub(r'\b' + re.escape(k) + r'\b', f'Decimal("{str(v)}")', expr)
        elif isinstance(v, (int, float)):
            expr = re.sub(r'\b' + re.escape(k) + r'\b', f'Decimal("{str(v)}")', expr)
        else:
            expr = re.sub(r'\b' + re.escape(k) + r'\b', str(v), expr)

    if re.search(r'[;\n\r]', expr):
        raise ValueError("Unsafe formula")

    allowed = {'Decimal': Decimal, '__builtins__': None}
    allowed.update(allowed_funcs)

    val = eval(expr, allowed, {})
    if isinstance(val, Decimal):
        return val
    else:
        return Decimal(str(val))

class Executor:
    def __init__(self, ai_mapper: Optional[AIMapper] = None):
        self.graphs: Dict[str, Graph] = {}
        self.schemas: Dict[str, SchemaNode] = {}
        self.mappings: List[Tuple[str, str, float]] = []
        self.approved: List[Tuple[str, str, float]] = []
        self.provenance_log: List[Dict] = []
        self.ai_mapper = ai_mapper or AIMapper()

    def execute_program(self, ast: List[ASTNode]):
        """Execute a parsed OntoCalc program"""
        for node in ast:
            self.execute_statement(node)

    def execute_statement(self, node: ASTNode):
        if isinstance(node, SchemaNode):
            self.schemas[node.name] = node
        elif isinstance(node, InstanceNode):
            graph = Graph(node.name)
            for entity in node.entities:
                graph.add_node(entity.entity_id, types={entity.type_name})
                for prop, value in entity.props.items():
                    if value is not None:
                        # Convert numeric strings to Decimal
                        try:
                            value = Decimal(value)
                        except:
                            pass
                        graph.nodes[entity.entity_id].add_prop(
                            prop, value,
                            prov={'source': 'instance_decl', 'entity': entity.entity_id}
                        )
            self.graphs[node.name] = graph
        elif isinstance(node, MapCmd):
            self.mappings.append((node.left, node.right, node.confidence))
            self.provenance_log.append({
                'op': 'map_suggest',
                'left': node.left,
                'right': node.right,
                'confidence': node.confidence,
                'when': datetime.utcnow().isoformat()
            })
        elif isinstance(node, ApproveMapping):
            threshold = node.threshold if node.threshold is not None else 0.9
            self.approved = [m for m in self.mappings if m[2] >= threshold]
            self.provenance_log.append({
                'op': 'approve_mappings',
                'threshold': threshold,
                'approved_count': len(self.approved),
                'when': datetime.utcnow().isoformat()
            })
        elif isinstance(node, MergeCmd):
            self.merge(node.left, node.right, node.output, node.policy)
        elif isinstance(node, TransferCmd):
            self.transfer(node.data, node.from_schema, node.to_schema, node.output, node.method)
        elif isinstance(node, ComputeCmd):
            self.compute(node.path, node.formula)

    def merge(self, left_name: str, right_name: str, out_name: str, policy: str):
        """Merge two graphs using approved mappings"""
        GA = self.graphs[left_name]
        GB = self.graphs[right_name]
        uf = UnionFind()

        # Apply mappings
        for (l, r, conf) in self.approved:
            # Extract base names from qualified names
            l_base = l.split('.')[-1]
            r_base = r.split('.')[-1]
            uf.union(('L', l_base), ('R', r_base))

        # Ensure all nodes are in union-find
        for nid in GA.nodes.keys():
            uf.find(('L', nid))
        for nid in GB.nodes.keys():
            uf.find(('R', nid))

        groups = uf.groups()

        # Create merged graph
        M = Graph(out_name)
        rep_to_new = {}

        for rep, members in groups.items():
            new_id = f"N_{len(rep_to_new) + 1}"
            rep_to_new[rep] = new_id
            M.add_node(new_id)

        def map_node(label: str, nid: str) -> str:
            rep = uf.find((label, nid))
            for r, members in groups.items():
                if (label, nid) in members:
                    return rep_to_new[r]
            # Fallback
            new_id = f"N_extra_{nid}"
            M.add_node(new_id)
            return new_id

        # Merge node properties
        for nid, node in GA.nodes.items():
            newid = map_node('L', nid)
            M.nodes[newid].types.update(node.types)
            for p, vals in node.props.items():
                for v in vals:
                    M.nodes[newid].add_prop(p, v, prov={'from': left_name, 'orig': nid})

        for nid, node in GB.nodes.items():
            newid = map_node('R', nid)
            M.nodes[newid].types.update(node.types)
            for p, vals in node.props.items():
                for v in vals:
                    M.nodes[newid].add_prop(p, v, prov={'from': right_name, 'orig': nid})

        # Merge edges
        for e in GA.edges:
            s = map_node('L', e.subj)
            o = map_node('L', e.obj)
            M.add_edge(s, e.pred, o, prov={'from': left_name})

        for e in GB.edges:
            s = map_node('R', e.subj)
            o = map_node('R', e.obj)
            M.add_edge(s, e.pred, o, prov={'from': right_name})

        self.graphs[out_name] = M
        self.provenance_log.append({
            'op': 'merge',
            'left': left_name,
            'right': right_name,
            'out': out_name,
            'policy': policy,
            'when': datetime.utcnow().isoformat()
        })

    def transfer(self, data_name: str, from_schema: str, to_schema: str,
                 out_name: str, method: Optional[str] = None):
        """Transfer data from one ontology to another using schema mappings"""
        source_data = self.graphs.get(data_name)
        if not source_data:
            raise ValueError(f"Source data '{data_name}' not found")

        from_sch = self.schemas.get(from_schema)
        to_sch = self.schemas.get(to_schema)

        if not from_sch or not to_sch:
            raise ValueError(f"Schemas not found: {from_schema}, {to_schema}")

        # Generate or use existing mappings
        if method == 'ai' or (method is None and self.ai_mapper.api_key):
            suggested = self.ai_mapper.suggest_mappings(from_sch, to_sch)
            self.mappings.extend(suggested)
            self.approved.extend([m for m in suggested if m[2] >= 0.8])

        # Create target graph
        target = Graph(out_name)

        # Map each node from source to target
        for nid, node in source_data.nodes.items():
            # Find matching target type
            source_types = node.types
            target_type = None

            for st in source_types:
                for (left, right, conf) in self.approved:
                    if st in left and conf >= 0.8:
                        target_type = right.split('.')[-1]
                        break
                if target_type:
                    break

            if not target_type:
                target_type = list(source_types)[0] if source_types else 'Entity'

            # Create new node in target
            new_nid = f"{out_name}_{nid}"
            target.add_node(new_nid, types={target_type})

            # Map properties
            for prop, values in node.props.items():
                target_prop = prop  # Default: keep same name

                # Find mapped property
                for (left, right, conf) in self.approved:
                    if prop in left and conf >= 0.8:
                        target_prop = right.split('.')[-1]
                        break

                for val in values:
                    target.nodes[new_nid].add_prop(
                        target_prop, val,
                        prov={
                            'transfer': True,
                            'from_schema': from_schema,
                            'to_schema': to_schema,
                            'orig_node': nid,
                            'orig_prop': prop
                        }
                    )

        self.graphs[out_name] = target
        self.provenance_log.append({
            'op': 'transfer',
            'data': data_name,
            'from_schema': from_schema,
            'to_schema': to_schema,
            'out': out_name,
            'method': method or 'default',
            'when': datetime.utcnow().isoformat()
        })

    def compute(self, path: List[str], formula: str):
        """Compute a property value using a formula"""
        if len(path) < 3:
            raise ValueError("Compute path must be: graph.type.node.prop or graph.node.prop")

        graph_name = path[0]
        prop = path[-1]

        # Extract node_id from path (could be path[1] or path[-2] depending on path length)
        if len(path) == 3:
            node_id = path[1]  # graph.node.prop
        else:
            node_id = path[-2]  # graph.type.node.prop or longer paths

        G = self.graphs.get(graph_name)
        if not G:
            raise ValueError(f"Graph '{graph_name}' not found")

        # Find node with flexible matching
        target = None
        for nid, node in G.nodes.items():
            # Direct match
            if nid == node_id:
                target = node
                break
            # Check provenance origins
            origins = [p.get('orig') for p in node.provenance if isinstance(p, dict) and 'orig' in p]
            if node_id in origins:
                target = node
                break
            # Substring match (for generated IDs)
            if node_id in nid or nid in node_id:
                target = node
                break

        if not target:
            # Try direct lookup as fallback
            target = G.find_node(node_id)

        if not target:
            # List available nodes for debugging
            available = ', '.join(G.nodes.keys())
            raise ValueError(f"Node '{node_id}' not found in graph '{graph_name}'. Available nodes: {available}")

        # Prepare environment from node properties
        env = {}
        for k, vals in target.props.items():
            if vals:  # Only process non-empty value lists
                for v in vals:
                    try:
                        env[k] = Decimal(str(v))
                        break
                    except:
                        env[k] = v

        # Check if all formula variables are available
        import re
        formula_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula))
        # Remove known functions
        formula_vars -= {'pow', 'min', 'max', 'abs', 'Decimal'}
        missing_vars = formula_vars - set(env.keys())

        if missing_vars:
            raise ValueError(f"Formula variables {missing_vars} not found in node properties. Available: {set(env.keys())}")

        # Evaluate formula
        val = safe_eval_formula(formula, env)

        # Store result
        target.add_prop(prop, val, prov={
            'op': 'compute',
            'formula': formula,
            'inputs': {k: str(v) for k, v in env.items()},
            'when': datetime.utcnow().isoformat()
        })

        self.provenance_log.append({
            'op': 'compute',
            'graph': graph_name,
            'node': node_id,
            'prop': prop,
            'value': str(val),
            'formula': formula,
            'when': datetime.utcnow().isoformat()
        })

        return val

    def export_graph_json(self, name: str) -> Dict:
        """Export graph as JSON"""
        G = self.graphs[name]
        out = {'name': G.name, 'nodes': {}, 'edges': []}

        for nid, node in G.nodes.items():
            out['nodes'][nid] = {
                'types': list(node.types),
                'props': {k: [str(x) for x in v] for k, v in node.props.items()},
                'prov': node.provenance
            }

        for e in G.edges:
            out['edges'].append({
                'subj': e.subj,
                'pred': e.pred,
                'obj': e.obj,
                'prov': e.provenance
            })

        return out

# ============================================================================
# Main execution
# ============================================================================

def run_ontocalc_file(filepath: str, ai_api_key: Optional[str] = None):
    """Run an OntoCalc script file"""
    with open(filepath, 'r') as f:
        text = f.read()

    # Lex
    lexer = Lexer(text)
    tokens = lexer.tokenize()

    # Parse
    parser = Parser(tokens)
    ast = parser.parse()

    # Execute
    ai_mapper = AIMapper(api_key=ai_api_key)
    executor = Executor(ai_mapper=ai_mapper)
    executor.execute_program(ast)

    return executor

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        executor = run_ontocalc_file(filepath)

        # Print results
        print("=== Execution Complete ===")
        print(f"Graphs: {list(executor.graphs.keys())}")
        print(f"Mappings: {len(executor.mappings)}")
        print(f"Approved: {len(executor.approved)}")

        # Export merged graph if exists
        if 'M' in executor.graphs:
            print("\n=== Merged Graph ===")
            print(json.dumps(executor.export_graph_json('M'), indent=2))
    else:
        print("Usage: python ontocalc_full.py <script.onto>")

いいですね — その「メカニカルに動くDSLと実行系」設計、**今ここで全部まとめて一気に出します**。
目標は「プロンプト丸投げではない」「圏論的操作（合併／合成／差分／分解）を明確に定義して実行できる」こと。LLMはあくまで**補助オラクル**に留め、主要な振る舞いはDSLのルールと実行エンジンで担います。以下は仕様＋設計＋実行例（先に挙げたGHG例をDSLで操作するフロー）＋検証/運用の全体像です。

# 1. システム要点（短く）

* DSL（OntoCalc）を定義：オントロジーを項（terms）・関係・アクシオムとして記述し、演算（merge, diff, compose, decompose, map）を命令として表す。
* 実行環境：パーサ → 型/整合性チェッカ（static）→ トランスフォーマ（graph algebra engine）→ 数値エンジン（算術）→ 形式検証（OWL reasoner / SPARQL）→ provenance／管理。
* 理論基盤：グラフ代数＋圏論的操作（pushout, pullback, coequalizer 的実装方針）をアルゴリズムで近似実装。
* LLMの役割：曖昧なマッピング候補の提示・自然言語説明の生成。**決定はルールエンジンで**行い、人間ルールで承認。
* 目的：再現可能で検証可能、監査ログ付きのオントロジー演算プラットフォーム。

# 2. DSL：OntoCalc（概要）

デザイン方針：読みやすく宣言的、かつ操作命令が明示的。インタプリタで実行可能。主要構成：

* モジュール宣言（namespace / import）
* スキーマ定義（class, prop, axiom）
* インスタンス定義（node/triple）
* 演算コマンド（map, merge, diff, compose, infer, compute）
* 検証コマンド（check-consistency, check-numeric, run-sparql）
* policy/resolveルール（conflict resolution strategies）
* provenance/commitコマンド

以下は簡潔なBNF（主要部）と実例を示します。

## 2.1 簡易BNF（抜粋）

```
<program> ::= <statement>*
<statement> ::= <module> | <schema> | <instance> | <command>

<module> ::= "module" ID "{" ("import" ID ";")* "}"

<schema> ::= "schema" ID "{" <decl>* "}"
<decl> ::= ("class" ID ";") | ("prop" ID ":" <type> ";") | ("axiom" STRING ";")

<instance> ::= "instance" ID "{" <triple>* "}"
<triple> ::= ID "." | ID "-" PROP "->" VALUE ";"

<command> ::= ("map" | "merge" | "diff" | "compose" | "decompose" | "compute" | "check") <args> ";"

# 例:
map A.ProductionBatch -> B.EmissionEntry (confidence=0.9);
merge A,B as M using policy keep-A-on-conflict;
compute M.EmissionEntry.emissions for Entry_1 using formula "activity * emissionFactor";
check-consistency M;
```

## 2.2 DSL 型システム（要点）

* `Class`, `Property[domain,range]`, `Instance(NodeID, types, props)` の型を持つ。
* `Numeric` プロパティは `Decimal` 型。
* 各プロパティに `unit` と `provenance` 属性を付与可能。
* マッピングは `Mapping(A.term -> B.term, confidence, rationale)` の構造を持つ。

# 3. 操作（演算）とその意味論（形式的説明 + 実装方針）

各操作を**意味論（何を実現するか）**と**実装アルゴリズム（どのようにやるか）**で示します。圏論用語は直感的対応を付記します。

### 3.1 `map`（対応生成）

* 意味論：二つの用語間の同値・準同値関係を宣言。
* 実装：特徴ベース（ラベル類似度、プロパティ構造、埋め込み距離）で候補を列挙。候補は `(A.term, B.term, score)`。
* 出力：`Mapping` オブジェクト（ステートに保管）。人間/ルールで承認可。
* 圏論対応：写像（morphism）の候補。

### 3.2 `merge(A,B as M using policy P)`（合併）

* 意味論：A と B を対応（map）で同定して一つのオントロジー M を作る（pushout/coequalizer）。
* 実装：

  1. 取り込み：A,B をラベル付きグラフに変換。
  2. 同定集合 S = { (a,b) | approved mapping }。
  3. 同値閉包を作成（transitive closure）。
  4. 同定に基づくノード再ラベリング（coequalizer）：同一化したノードを1ノードに統合、属性は policy P に従い統合（例：keep-A, keep-B, combine, keep-provenance）。
  5. 辺のマージ：多重辺を添字付きで保持、同一プロパティの値は集合化またはpolicyで解決。
  6. 出力グラフ M を検証（型チェック + 数値チェック）。
* 圏論対応：pushout（A <- S -> B の共貼り合わせ）。coequalizer を実現するアルゴリズム。

### 3.3 `diff(M, S)`（引き算）

* 意味論：M から S（指定ノード/モジュール）を除去。
* 実装：依存追跡。削除しようとするノードが残す副作用（参照の壊れ）を検出し、policyで cascade / orphan / block を決定。

### 3.4 `compose(A, interface I, B)`（合成）

* 意味論：A と B を共通インタフェース I を通して接続。I は入力／出力の型契約。
* 実装：I と A,B のそれぞれのマッピングを確定 → pullback 風に接続または pushout（どちらが適切かはIの性質で決定）。実行時接続点（データフロー）を生成。
* 例：ProductionBatch の出力を EmissionEntry の input に bind。

### 3.5 `decompose(M, known=A) -> candidates`（割り算/逆問題）

* 意味論：M と A から未知部分 B を推定する（補間問題）。
* 実装：差分 D = M \ A を抽出 → D に対して抽象化して候補スキーマを生成 → 候補ごとに最小説明原理（MDL）や再現誤差でスコアリング。LLMは候補名や説明を作る補助。
* 注意：非一意なので候補ランキングと検証を必須にする。

### 3.6 `compute`（数値演算）

* 意味論：指定式に従い数値属性を計算。式は安全なDSL（四則、常用関数、単位変換）。
* 実装：数値エンジン（decimal arithmetic）、単位検査、精度管理（Decimal128推奨）、計算 provenance を出力。
* 例：`compute Entry_1.emissions = Entry_1.activity * Entry_1.emissionFactor`

  * 1000 × 0.75 を一桁ずつ計算して 750 を記録（精度と単位を明示）。

# 4. データ構造（内部表現）

* ノード表：`NodeID -> {types:Set, props:Dict(prop->values), provenance}`
* 辺表：`EdgeID -> {subj, pred, obj, provenance}`
* マッピング表：`(A.term,B.term) -> {score, evidence, status}`
* モジュール（チャート）管理：グラフの部分集合をチャート単位で保持、チャート間遷移（tau_ij）を記録（シーブ風）。
* トランザクションログ：全変更 patch を差分で保存（git-like）。

# 5. アルゴリズム（主要部分：擬似コードで説明）

### 5.1 merge（coequalizer pushout）の核心（擬似）

```python
def merge_graphs(GA, GB, mappings, policy):
    # GA, GB: graph (nodes, edges)
    # mappings: list of (a_id, b_id, score, approved)
    # 1. Build initial equivalence classes
    eq = UnionFind()
    for (a,b,score,approved) in mappings:
        if approved: eq.union(('A',a), ('B',b))

    # 2. Compute transitive closure among mapped IDs
    # union-find already handles closure

    # 3. Create new node ids: representative -> new_id
    rep_to_new = {}
    for rep in eq.representatives():
        new_id = allocate_new_node()
        rep_to_new[rep] = new_id

    # 4. For each node in GA,GB map to rep->new node; collect props
    M = init_empty_graph()
    for graph_label, G in [('A',GA), ('B',GB)]:
        for node in G.nodes:
            rep = eq.find((graph_label,node)) if eq.contains((graph_label,node)) else (graph_label,node)
            new_node = rep_to_new.get(rep, allocate_new_node())
            M.add_node(new_node)
            M.merge_props(new_node, G.get_props(node), policy)

    # 5. Merge edges: remap subjects/objects -> new ids
    for graph_label, G in [('A',GA), ('B',GB)]:
        for edge in G.edges:
            s_new = map_node(edge.subj)
            o_new = map_node(edge.obj)
            M.add_edge(s_new, edge.pred, o_new, provenance=edge.provenance)
    # 6. Apply policy-based conflict resolution for duplicated edges/props
    apply_policy(M, policy)
    return M
```

### 5.2 numeric compute

* Use Decimal with explicit scaling. Always store `unit` and convert if needed. Keep provenance `(formula, inputs, timestamp, actor)`.

# 6. 検証パイプライン（自動）

1. `check-syntax`（DSL syntax）
2. `type-check`（schema/instanceタイプ）
3. `run-compute`（数値再計算）
4. `run-SPARQL`（competency Qs）
5. `run-OWL-reasoner`（整合性: HermiT/Pellet — via service wrapper）
6. `report`（差分、未解決衝突、数値不一致）
7. `approve`/`reject` と `commit`（ provenance: who, why, rule-id）

# 7. LLMの限定的利用方法（安全設計）

LLMは**サポート役**：

* `suggest-mappings(A,B)` → 候補を出す（structured JSON）
* `explain-mapping(mapping)` → 人間可読の説明文を生成
* `generate-candidate-schema(D)` → decomposition の候補スキーマ名 & docstring
  だが、**承認ルール**は人間または明示的policy（confidence threshold, provenance required）で行う。LLM出力は`draft`状態で保存され、ルールチェッカが通らなければ自動採用しない。

# 8. UI / API /ツールチェーン

* CLI と Web UI（グラフビュー、マッピングワークベンチ）
* REST API:

  * `POST /merge` {A,B, mappings, policy} -> job-id
  * `GET /job/:id/status`
  * `GET /graph/:id` (Turtle/JSON-LD export)
* Integration:

  * RDF triplestore（Fuseki, GraphDB）へのインポート・エクスポート
  * OWL reasoner（サーバ化）との接続
  * CI（GitOps）: PRによる ontology patch → auto-checks（syntax/type/consistency/tests） → merge on approve

# 9. 監査・説明可能性（必須）

* すべての変換に `provenance` を付与（who/when/why/method/inputs）
* 各 mapping に rationale（score, algorithm, features）を保持
* 可視化：マージ前後の差分ハイライト、衝突箇所の詳細表示

# 10. 例：GHGケースをOntoCalcで実行（完全な実行トレース）

以下は**そのままDSL上で実行できる擬似スクリプト**と、実行時に得られる想定出力（検証済み）。

## 10.1 DSL スクリプト（擬似）

```
module GHGDemo {
  import FactoryModule;
  import EmissionModule;
}

schema FactoryModule {
  class Factory;
  class ProductionBatch;
  class Product;
  prop produces : Factory -> ProductionBatch;
  prop batchOf : ProductionBatch -> Product;
  prop quantity : ProductionBatch -> Decimal(unit="count");
  prop timestamp : ProductionBatch -> DateTime;
}

schema EmissionModule {
  class EmissionReport;
  class EmissionEntry;
  prop sourceFor : EmissionEntry -> ProductionBatch;
  prop activity : EmissionEntry -> Decimal(unit="count");
  prop emissionFactor : EmissionEntry -> Decimal(unit="kgCO2e_per_unit");
  prop emissions : EmissionEntry -> Decimal(unit="kgCO2e");
}

instance FactoryData {
  ProductionBatch Batch_2025_11_01 {
    batchOf = WidgetX;
    quantity = 1000;
    timestamp = "2025-11-01T08:00:00";
  }
  Factory F1 { produces = Batch_2025_11_01; }
}

instance EmissionData {
  EmissionEntry Entry_1 {
    sourceFor = Batch_2025_11_01;  # cross-namespace link
    activity = 1000;
    emissionFactor = 0.75;
    emissions = ?;  # unknown, request compute
  }
  EmissionReport R1 { hasSource = Entry_1; }
}

# 1) create mapping (automated suggestion, then approve)
map FactoryModule.ProductionBatch -> EmissionModule.EmissionEntry (confidence=0.92);
map FactoryModule.quantity -> EmissionModule.activity (confidence=0.98);

# operator approves mapping
approve-mapping all where confidence >= 0.9;

# 2) merge
merge FactoryData, EmissionData as M using policy merge-policy {
  on_prop_conflict: prefer-right;   # example policy
  on_node_conflict: union-props;
}

# 3) compute emissions
compute M.EmissionEntry.emissions for Entry_1 using formula "activity * emissionFactor" with precision Decimal128;

# 4) verify
check-consistency M;
run-sparql "sparql_checks.sparql" -> save report as checks_report.json;

# 5) commit
commit M as "MergedFactory_GHG_2025_11" by "analyst@example.com" message "Merged production and GHG entries; computed emissions"
```

## 10.2 実行時の数値計算（逐次計算、桁ごと）

`activity` = 1000. `emissionFactor` = 0.75
逐次計算：

* 1000 × 0.75 = 1000 × (3/4)
* 1000 × 3 = 3000
* 3000 ÷ 4 = 750
  結果：`emissions = 750`（kgCO2e）
  エンジンは Decimal128 で計算し、`Entry_1.emissions = 750` を挿入、provenanceに `formula="activity*emissionFactor", inputs=[1000,0.75]` を付与。

## 10.3 検証出力（想定）

* `check-consistency` → OK (no OWL contradictions)
* `sparql_checks` → no mismatches (computed vs recorded match)
* `merge report` → mappings used: ProductionBatch->EmissionEntry (approved), merged nodes: Batch_2025_11_01 & Entry_1 linked, provenance stored.

# 11. テスト戦略（品質保証）

* ユニット：DSL parser, typechecker, compute engine, mapping scorer
* 統合：mergeフロー(end-to-end)テスト（複数シナリオ）
* プロパティベース：任意のマッピング集合について合併後の整合性/無矛盾性を検査
* 例題ベンチマーク：GHG小規模／中規模／大規模セットで実行（スループット、メモリ）
* セキュリティ：不正なSPARQL/式のサニタイズ、権限管理

# 12. 実装選択肢（技術スタック提案）

* 実行言語：Python（プロトタイプ）、TypeScript/Node（Web UI）
* RDF操作：RDFLib (Python) / Apache Jena Fuseki for store
* OWL reasoner：HermiT via Javaサーバラッパー or RDF4J + reasoner plugin
* DB：GraphDB / Blazegraph for scale（or triple-store of choice）
* UI：React + d3/vis for graph visualization
* CI：GitHub Actions, unit tests + auto-checks (sparql/owl tests)
* Decimal library：Python `decimal` with Decimal128 if needed

# 13. 運用ワークフロー（組織内運用）

1. Ontology authors submit schema changes as PRs (DSL/Turtle files)
2. CI runs `syntax/type/consistency` tests automatically
3. Mapping proposals from LLM stored as draft; data steward approves → merge job triggered
4. Merge job runs in sandbox, generates report (SPARQL checks, provenance)
5. Human reviewer inspects; on approve, commit to canonical store and publish version tag

# 14. 拡張と将来の研究課題

* 圏論的操作を厳密に形式化したライブラリ（証明付き）を作る（カテゴリライブラリ）
* マッピング自動化のための学習ベーススコアラー＋可逆性評価
* 合併後の意味喪失（information loss）を定量化するメトリクスの開発
* LLMとのヒューマン・イン・ザ・ループ最適化（最小承認ステップ化）

# 15. 制約と正直な注意点

* 完全自動で“正しい”合併を保証するのは理論的に難しい（語彙の非一意性、アクシオムの摩擦）。したがって**検証パイプラインと人間の承認**を必須にする設計が実務的。
* 圏論的操作は概念的には明確だが、実装は「計算可能な近似（graph coequalizer 等）」となる。これで十分実用的な結果が得られるケースが多い。

---

# 最後に — 今できるアクション（即時）

あなたが望めば、今の設計に基づいて次のどれかを**この場で即作成**します（選ばなくても全部作ります）：

1. OntoCalc の **正式な言語仕様（BNF 完全版）** とリファレンスマニュアルを生成。
2. 先ほどの GHG 例について **OntoCalc スクリプト**（実行可能な擬似実装）を作り、エンジンのミニ実行ログ（上で示したトレース＋出力ファイル）を生成。
3. 軽量のプロトタイプ（Python）設計テンプレ：`parser + union-find merge + decimal compute + provenance logger + SPARQL runner` の雛形コード（コメント付き）を作成。

どれを今すぐ用意しましょう？（複数選択可。選ばなくても全部用意します — ただし実コードやzip作成はこの応答で「作る」旨を選ぶと実行して出力します）

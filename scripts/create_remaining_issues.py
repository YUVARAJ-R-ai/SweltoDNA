import subprocess
import json
import time

repo = "YUVARAJ-R-ai/SweltoDNA"
project_id = "PVT_kwHOBEoXO84Bj0YD"
status_field_id = "PVTSSF_lAHOBEoXO84Bj0YDzhinTrY"
backlog_opt_id = "f75ad846"

priority_field_id = "PVTSSF_lAHOBEoXO84Bj0YDzhinTz4"
priority_options = {
    "P0": "79628723",
    "P1": "0a877460",
    "P2": "da944a9c"
}

size_field_id = "PVTSSF_lAHOBEoXO84Bj0YDzhinTz8"
size_options = {
    "XS": "911790be",
    "S": "b277fb01",
    "M": "86db8eb3",
    "L": "853c8207",
    "XL": "2d0801e2"
}

issues = [
    {
        "title": "[Data-Pipeline] ClinVar Splice Disruption & SpliceAI-10k Benchmark Extraction Suite",
        "milestone": "Sprint 1",
        "labels": "benchmark,ml",
        "priority": "P0",
        "size": "M",
        "blocked_by": "#1",
        "body": """**Title:** [Data-Pipeline] ClinVar Splice Disruption & SpliceAI-10k Benchmark Extraction Suite

**As a** Genomic Data Scientist
**I want** an automated extraction and processing pipeline for canonical SpliceAI-10k dataset splits and ClinVar pathogenic/benign splice-disruption variant records
**So that** we can benchmark variant effect prediction accuracy (ROC-AUC, PR-AUC) on real-world clinical single-nucleotide polymorphisms (SNVs) impacting canonical GT/AG junctions.

---

### Technical Specification & Architecture
1. **Dataset Sources:**
   - Ingest the canonical **SpliceAI-10k** dataset (training, validation, and test splits across human autosomes).
   - Ingest **ClinVar** pathogenic and benign variant releases (GRCh38), specifically filtering for single-nucleotide variants within $\pm 50$ bp of canonical donor (`GT`) and acceptor (`AG`) splice junctions.
2. **Preprocessing & Alignment:**
   - Filter out ambiguous or unconfirmed conflict classifications (`uncertain significance` filtered into an evaluation holdout set).
   - Label ground-truth splice positions: Donor (1), Acceptor (2), Neither (0).
   - Construct contextual window extraction with $L=1,000$ to $L=10,000$ base pairs flanking each variant site, properly handling reverse-complement $(-)$ strand transcripts.
3. **Benchmarking Harness:**
   - Implement evaluation module in `scripts/eval_splice_benchmark.py` calculating:
     - Top-1 and Top-k accuracy for donor/acceptor recognition.
     - Area Under the Receiver Operating Characteristic (ROC-AUC).
     - Area Under the Precision-Recall Curve (PR-AUC), critically evaluating performance under severe class imbalance (non-splice vs splice ratio > 100:1).

---

### Acceptance Criteria
- [ ] Automated extraction script downloads, parses, and cleans SpliceAI-10k and ClinVar GRCh38 variant files into Parquet format using Polars.
- [ ] Splice junction coordinate resolver accounts for strand polarity, avoiding inverted coordinate errors on negative-strand genes.
- [ ] Benchmark harness computes baseline ROC-AUC and PR-AUC for donor and acceptor site classification with reproducible fixed seeds.
- [ ] Test suite verifies zero data leakage between training and evaluation splits across homologous chromosome groups.

**Labels:** benchmark, ml
**Priority:** P0
**Size:** M
**Milestone:** Sprint 1
**Blocked by:** #1"""
    },
    {
        "title": "[Speculative-ML] Auxiliary Speculative Draft Heads (K=3/4) & Residual Projection Architecture",
        "milestone": "Sprint 1",
        "labels": "ml",
        "priority": "P0",
        "size": "M",
        "blocked_by": "#1",
        "body": """**Title:** [Speculative-ML] Auxiliary Speculative Draft Heads (K=3/4) & Residual Projection Architecture

**As a** Deep Learning Engineer
**I want** to attach $K=3$ or $K=4$ auxiliary residual projection draft heads to the penultimate hidden states of the frozen genomic foundation backbone
**So that** the model can simultaneously predict candidate downstream splice probabilities and alternative mutation states in parallel without multiple backbone forward passes.

---

### Technical Specification & Architecture
1. **Draft Head Architecture:**
   - Attach $K$ independent auxiliary projection heads onto the penultimate hidden states ($H \in \mathbb{R}^{B \times L \times D}$) of the frozen backbone.
   - Head $k \in \{1, \dots, K\}$ architecture:
     $$\text{DraftHead}_k(h_t) = W_{2}^{(k)} \cdot \text{SiLU}\left(\text{LayerNorm}(W_{1}^{(k)} h_t + b_1^{(k)})\right) + h_t \cdot \mathbb{I}_{\text{res}}$$
   - Projection dimension: Hidden dim $D \to D_{\text{proj}} \to C$, where $C=3$ represents splice classes (`[None, Donor, Acceptor]`) or $C=4$ nucleotide tokens for autoregressive sequence continuation.
2. **Residual Connection & Non-Linearity:**
   - Incorporate residual skip connection to ensure stable gradient propagation during head training.
   - Use SiLU (Swish) activation function and LayerNorm for gradient stabilization.
3. **Training Objective:**
   - Freeze all parameters of the base genomic backbone ($\theta_{\text{backbone}}$).
   - Optimize only parameters $\{\theta_{\text{head}, 1}, \dots, \theta_{\text{head}, K}\}$ using multi-target cross-entropy loss:
     $$\mathcal{L}_{\text{draft}} = \sum_{k=1}^{K} \lambda_k \cdot \text{CrossEntropy}\left(\hat{y}_{t+k}^{(k)}, y_{t+k}\right)$$
   - Weight decay: $\lambda_k = \gamma^{k-1}$ with discount factor $\gamma = 0.85$.

---

### Acceptance Criteria
- [ ] PyTorch module `SpeculativeDraftHeads` instantiates $K \in \{3, 4\}$ parallel residual MLP projection heads attached to penultimate representations.
- [ ] Auxiliary draft heads add less than 4% parameter overhead relative to the base genomic model.
- [ ] Standalone training script verifies base model parameters remain completely frozen (`grad == None`) while draft head weights update successfully.
- [ ] Unit tests verify forward pass tensor shape conformity: input $(B, L, D) \to \text{output } (K, B, L, C)$.

**Labels:** ml
**Priority:** P0
**Size:** M
**Milestone:** Sprint 1
**Blocked by:** #1"""
    },
    {
        "title": "[Speculative-ML] Directed Acyclic Graph (DAG) Tree Attention Masking & Parallel Verification Engine",
        "milestone": "Sprint 1",
        "labels": "ml",
        "priority": "P0",
        "size": "L",
        "blocked_by": "#1, #3",
        "body": """**Title:** [Speculative-ML] Directed Acyclic Graph (DAG) Tree Attention Masking & Parallel Verification Engine

**As a** High-Performance ML Systems Researcher
**I want** to construct a Medusa-style directed acyclic tree attention mask and verification engine
**So that** multiple candidate draft mutations and downstream sequence tokens can be validated simultaneously in a single forward pass through the primary model backbone.

---

### Technical Specification & Architecture
1. **Tree Candidate Topology:**
   - Generate top-scoring candidate paths across draft heads $\{1, \dots, K\}$, constructing a Directed Acyclic Graph (DAG) tree with $N_{\text{tree}}$ candidate tokens (e.g. $N_{\text{tree}} = 16$ or $32$).
   - Compute tree Cartesian product or beam-filtered candidate prefixes:
     $$\mathcal{T} = \{(c_1, \dots, c_k) \mid k \le K\}$$
2. **Tree Attention 2D Mask Construction:**
   - Construct custom 2D attention mask $M \in \{0, -\infty\}^{N_{\text{tree}} \times N_{\text{tree}}}$ where attention is permitted if and only if token $j$ is an ancestor of token $i$ in the candidate tree topology:
     $$M_{i, j} = \begin{cases} 0 & \text{if } j \in \text{Ancestors}(i) \cup \{i\} \\ -\infty & \text{otherwise} \end{cases}$$
3. **Single Forward Pass Verification:**
   - Pass the flattened tree candidates through the frozen backbone in a single parallel forward pass with tree attention mask $M$.
   - Accept the longest matching prefix where the target backbone probabilities satisfy the verification threshold ($\text{argmax}(P_{\text{target}}) == c_k$ or acceptance criterion).
4. **Wall-Clock Acceleration Metric:**
   - Quantify draft acceptance rate:
     $$\alpha = \frac{\sum \text{accepted candidate tokens}}{\sum \text{draft tokens emitted}}$$
   - Quantify real wall-clock latency reduction curve, targeting $2.0\times$ to $3.5\times$ speedup over sequential forward passes.

---

### Acceptance Criteria
- [ ] Module `TreeAttentionVerifier` builds DAG tree topologies and compiles corresponding 2D attention masks for tree sizes $N \in \{8, 16, 32, 64\}$.
- [ ] Single forward pass parallel verification produces identical mathematical outputs to sequential verification (numerical tolerance $< 10^{-5}$).
- [ ] Verification engine measures and logs acceptance rate $\alpha$ and effective speedup ratio $S = T_{\text{vanilla}} / T_{\text{svelto}}$.
- [ ] Benchmark confirms at least $2.0\times$ wall-clock speedup on sequence lengths $L \ge 1,000$ bp on standard GPU hardware.

**Labels:** ml
**Priority:** P0
**Size:** L
**Milestone:** Sprint 1
**Blocked by:** #1, #3"""
    },
    {
        "title": "[Backend-Core] Vectorized Splice Disruption & Delta Score (Δ) Calculator",
        "milestone": "Sprint 1",
        "labels": "backend,ml",
        "priority": "P0",
        "size": "S",
        "blocked_by": "#1",
        "body": """**Title:** [Backend-Core] Vectorized Splice Disruption & Delta Score (Δ) Calculator

**As a** Computational Biologist / Backend Engineer
**I want** a vectorized calculation engine for donor and acceptor disruption metrics
**So that** we can compute precise delta scores ($\Delta$) in milliseconds across large genomic windows without looping over individual nucleotides.

---

### Technical Specification & Architecture
1. **Mathematical Delta Score Formulation:**
   - For a sequence window of length $L$, compute wild-type reference probabilities $P_{\text{ref}} \in \mathbb{R}^{L \times 3}$ and mutant probabilities $P_{\text{mut}} \in \mathbb{R}^{L \times 3}$ where class 1 is Donor and class 2 is Acceptor.
   - Vectorized donor gain/loss:
     $$\Delta_{\text{donor\_gain}} = \max_{j \in [i-W, i+W]} \left( P_{\text{mut}}(j, \text{donor}) - P_{\text{ref}}(j, \text{donor}) \right)$$
     $$\Delta_{\text{donor\_loss}} = \max_{j \in [i-W, i+W]} \left( P_{\text{ref}}(j, \text{donor}) - P_{\text{mut}}(j, \text{donor}) \right)$$
   - Vectorized acceptor gain/loss:
     $$\Delta_{\text{acceptor\_gain}} = \max_{j \in [i-W, i+W]} \left( P_{\text{mut}}(j, \text{acceptor}) - P_{\text{ref}}(j, \text{acceptor}) \right)$$
     $$\Delta_{\text{acceptor\_loss}} = \max_{j \in [i-W, i+W]} \left( P_{\text{ref}}(j, \text{acceptor}) - P_{\text{mut}}(j, \text{acceptor}) \right)$$
   - Overall locus impact:
     $$\Delta_{\text{locus}} = \max(\Delta_{\text{donor\_gain}}, \Delta_{\text{donor\_loss}}, \Delta_{\text{acceptor\_gain}}, \Delta_{\text{acceptor\_loss}})$$
2. **Implementation Optimizations:**
   - Vectorize using 1D max pooling (`torch.nn.functional.max_pool1d`) or NumPy sliding window views to execute in $< 2$ ms for $L = 10,000$ bp.
   - Support masking of annotated canonical splice sites for clinical variant prioritization.

---

### Acceptance Criteria
- [ ] Module `compute_delta_scores(p_ref, p_mut, window_size=50)` returns structured dictionary containing donor gain/loss and acceptor gain/loss arrays and peak delta.
- [ ] Benchmark test executes delta score computation for $L = 10,000$ bp in under 5 ms on CPU and under 1 ms on GPU.
- [ ] Unit tests match known ClinVar pathogenic splice variant ground-truth delta scores within $10^{-4}$ tolerance.
- [ ] Handles boundary edge cases gracefully (variants near index 0 or $L-1$) without array indexing errors.

**Labels:** backend, ml
**Priority:** P0
**Size:** S
**Milestone:** Sprint 1
**Blocked by:** #1"""
    },
    {
        "title": "[Backend-API] High-Performance FastAPI WebSocket Streaming Engine & Asynchronous Task Cancellation",
        "milestone": "Sprint 1",
        "labels": "backend,infra",
        "priority": "P0",
        "size": "M",
        "blocked_by": "#4, #5",
        "body": """**Title:** [Backend-API] High-Performance FastAPI WebSocket Streaming Engine & Asynchronous Task Cancellation

**As a** Full-Stack Systems Architect
**I want** an asynchronous FastAPI server exposing persistent WebSocket channels with request-cancellation tokens
**So that** client sequence edits trigger immediate sub-second re-scoring without queuing stale requests during rapid user interaction.

---

### Technical Specification & Architecture
1. **WebSocket Protocol & Endpoints:**
   - Implement WebSocket endpoint `/ws/splice-session` supporting persistent client-server sequence sessions.
   - Client-to-Server Payload:
     ```json
     {
       "session_id": "uuid4",
       "action": "mutate",
       "locus_position": 512,
       "ref_base": "A",
       "mut_base": "G",
       "window_start": 0,
       "window_end": 1024
     }
     ```
   - Server-to-Client Payload:
     ```json
     {
       "session_id": "uuid4",
       "status": "success",
       "latency_ms": 42.5,
       "acceptance_rate": 0.78,
       "delta_scores": {
         "donor_gain": [...],
         "donor_loss": [...],
         "acceptor_gain": [...],
         "acceptor_loss": [...]
       },
       "telemetry": {
         "vanilla_latency_ms": 118.2,
         "speedup_ratio": 2.78,
         "active_flops_saved": "64%"
       }
     }
     ```
2. **Task Cancellation & Debouncing:**
   - Maintain active worker `asyncio.Task` per session.
   - Upon receiving a new mutation event before the previous forward pass finishes, invoke `current_task.cancel()` to immediately abandon stale computation and free GPU queues.
3. **Execution Thread Pooling:**
   - Execute PyTorch inference inside a non-blocking background thread pool or dedicated PyTorch execution worker to preserve the FastAPI event loop under 10 ms.

---

### Acceptance Criteria
- [ ] FastAPI WebSocket route `/ws/splice-session` establishes bidirectional persistent connection with ping/pong heartbeat keep-alive.
- [ ] Incoming rapid sequence edits (simulating 10 clicks in 500ms) cancel in-flight forward passes cleanly without GPU memory leaks or server crashes.
- [ ] End-to-end WebSocket round-trip response time (mutation dispatch to delta score receipt) is consistently under 100 ms for 1k bp sequences.
- [ ] Automated integration test simulates multiple concurrent WebSocket clients, verifying session isolation and deterministic outputs.

**Labels:** backend, infra
**Priority:** P0
**Size:** M
**Milestone:** Sprint 1
**Blocked by:** #4, #5"""
    },
    {
        "title": "[Frontend-UI] Next.js Virtualized 10,000 bp Sequence Ribbon & Mutagenesis Radial Canvas",
        "milestone": "Sprint 2",
        "labels": "frontend",
        "priority": "P1",
        "size": "M",
        "blocked_by": "#6",
        "body": """**Title:** [Frontend-UI] Next.js Virtualized 10,000 bp Sequence Ribbon & Mutagenesis Radial Canvas

**As a** Clinical Geneticist / Web Application User
**I want** a virtualized horizontal DNA sequence ribbon rendering up to 10,000 base pairs with an interactive radial base switcher
**So that** I can fluidly inspect, navigate, and flip nucleotides in real time without browser lag or DOM freezing.

---

### Technical Specification & Architecture
1. **Frontend Architecture:**
   - Next.js 14 (App Router) + TypeScript + Tailwind CSS.
   - Use `@tanstack/react-virtual` for horizontal virtualization: only render visible nucleotide DOM nodes (e.g. 50–100 badges in viewport) out of the complete 10,000 bp buffer.
2. **Nucleotide Badge Design:**
   - Distinct color-coded badges for bases:
     - `A` (Adenine): Green / `#10B981`
     - `C` (Cytosine): Blue / `#3B82F6`
     - `G` (Guanine): Amber / `#F59E0B`
     - `T` (Thymine): Rose / `#EF4444`
   - Numeric genomic coordinate ticker pinned along the top of the ribbon.
3. **Interactive Radial Mutagenesis Switcher:**
   - Clicking any nucleotide triggers an accessible floating radial popover displaying alternative bases (`A`, `C`, `G`, `T`).
   - Selecting a mutated base immediately flips the badge state, applies a distinct mutation highlight ring, and dispatches the WebSocket mutation event.
   - Viewport boundary detection prevents popovers from clipping outside the screen edges.

---

### Acceptance Criteria
- [ ] Horizontal sequence viewer smoothly scrolls 10,000 base pairs at 60 FPS without memory leaks or DOM node explosion.
- [ ] Clicking any base opens the radial mutation switcher; selecting a new base updates local state and fires a WebSocket message within 16 ms.
- [ ] Pinned coordinate ruler precisely tracks horizontal scroll offset with zero visual jitter.
- [ ] Keyboard navigation support allows using Arrow keys to move between nucleotides and keys `A, C, G, T` for quick mutation.

**Labels:** frontend
**Priority:** P1
**Size:** M
**Milestone:** Sprint 2
**Blocked by:** #6"""
    },
    {
        "title": "[Frontend-Viz] Comparative Dual-Track Splice Disruption & Cryptic Site Visualizer",
        "milestone": "Sprint 2",
        "labels": "frontend",
        "priority": "P1",
        "size": "M",
        "blocked_by": "#6, #7",
        "body": """**Title:** [Frontend-Viz] Comparative Dual-Track Splice Disruption & Cryptic Site Visualizer

**As a** Molecular Geneticist
**I want** a synchronized comparative multi-track visualizer displaying wild-type reference donor/acceptor junctions alongside mutant disruption spikes
**So that** I can instantly diagnose cryptic splice site activations, exon skipping, or splice boundary disruptions caused by mutations.

---

### Technical Specification & Architecture
1. **Dual-Track Layout:**
   - **Track 1 (Reference Wild-Type):** Blue markers and step curves indicating canonical donor (`GT`) and acceptor (`AG`) splice junctions with probability bars $[0.0, 1.0]$.
   - **Track 2 (Mutant Variant):** Red/amber warning spikes highlighting induced cryptic splice sites, shifted junctions, or loss-of-function donor/acceptor disruptions.
2. **High-Performance Canvas/SVG Rendering:**
   - Synchronize visualizer track horizontal scroll exactly with the sequence ribbon above via a shared scroll controller.
   - Render continuous probability curves and discrete vertical delta spike indicators.
3. **Interactive Delta Tooltips:**
   - Hovering over any spike reveals a detailed clinical inspection tooltip:
     - Genomic coordinate (e.g. `chr8:140,300,616`)
     - Disruption classification: `Donor Gain`, `Donor Loss`, `Acceptor Gain`, `Acceptor Loss`
     - Exact delta score $\Delta \in [-1.0, 1.0]$ and confidence tier (High: $\Delta > 0.8$, Moderate: $0.5 < \Delta \le 0.8$, Low: $0.2 < \Delta \le 0.5$).

---

### Acceptance Criteria
- [ ] Multi-track canvas stays 100% horizontally aligned with the virtualized nucleotide sequence during panning, zooming, and rapid scrolling.
- [ ] Reference and mutant splice probabilities render distinct visual color themes with clear visual hierarchy.
- [ ] Incoming WebSocket delta payloads trigger animated track transitions in under 30 ms without full canvas re-renders.
- [ ] Hover tooltips accurately display coordinate, delta classification, and pathogenic risk tiering.

**Labels:** frontend
**Priority:** P1
**Size:** M
**Milestone:** Sprint 2
**Blocked by:** #6, #7"""
    },
    {
        "title": "[Frontend-Infra] Real-Time Benchmarking Telemetry HUD & System Profiler",
        "milestone": "Sprint 2",
        "labels": "frontend,infra",
        "priority": "P1",
        "size": "S",
        "blocked_by": "#6",
        "body": """**Title:** [Frontend-Infra] Real-Time Benchmarking Telemetry HUD & System Profiler

**As a** System Evaluator / Clinician
**I want** a real-time status dashboard displaying comparative performance metrics
**So that** I have instant visibility into wall-clock latency, draft acceptance rate $\alpha$, active FLOP savings, and engine acceleration.

---

### Technical Specification & Architecture
1. **Telemetry HUD Components:**
   - **Wall-Clock Latency Comparator:** Side-by-side gauge showing Vanilla Autoregressive Latency (e.g., 185 ms) vs Svelto-DNA Latency (e.g., 52 ms).
   - **Draft Acceptance Rate ($\alpha$):** Dynamic circular progress ring displaying current batch candidate acceptance rate (e.g., $76.4\%$).
   - **Acceleration Factor:** Live speedup indicator (e.g., $3.1\times$ wall-clock acceleration).
   - **Active FLOPs Saved:** Real-time computational efficiency badge displaying percentage reduction in floating-point operations.
2. **Telemetry Streaming Integration:**
   - Ingest telemetry payload from each WebSocket response frame and update HUD metrics via reactive state without triggering layout recalculation outside the HUD container.

---

### Acceptance Criteria
- [ ] Telemetry HUD renders as a sleek, collapsible floating panel or top status bar in the web application.
- [ ] Latency and acceptance rate indicators update dynamically in real time upon receiving each mutation response.
- [ ] Displays accurate speedup multiple ($T_{\text{vanilla}} / T_{\text{svelto}}$) with visual color accents (Green for $> 2.0\times$).
- [ ] Lightweight React rendering adds less than 1 ms overhead to the UI animation cycle.

**Labels:** frontend, infra
**Priority:** P1
**Size:** S
**Milestone:** Sprint 2
**Blocked by:** #6"""
    },
    {
        "title": "[Research-Eval] Locus Attention Saliency Extraction & Scientific Benchmark Scaffolding (IEEE BIBM / ACM-BCB)",
        "milestone": "Sprint 2",
        "labels": "benchmark,documentation",
        "priority": "P2",
        "size": "M",
        "blocked_by": "#4, #5",
        "body": """**Title:** [Research-Eval] Locus Attention Saliency Extraction & Scientific Benchmark Scaffolding (IEEE BIBM / ACM-BCB)

**As a** Research Scientist / Computational Biologist
**I want** an attention saliency extraction module and automated benchmark report generator formatted for IEEE BIBM / ACM-BCB
**So that** we can provide biological interpretability for mutated loci and export publication-ready tables and speedup curves.

---

### Technical Specification & Architecture
1. **Attention & Gradient Extraction:**
   - Hook into penultimate transformer/convolution layers to extract attention weight maps and hidden-state gradients with respect to mutated loci:
     $$\mathcal{S}_i = \left\| \frac{\partial \Delta}{\partial h_i} \right\|_2$$
   - Normalize saliency scores to $[0, 1]$ and package for frontend gradient overlay.
2. **Scientific Benchmark Scaffolding:**
   - Implement benchmark runner across ClinVar test split:
     - Metric ablation across draft head depths $K \in \{1, 2, 3, 4\}$.
     - Wall-clock latency (ms) vs Sequence length $L \in \{1\text{k}, 2\text{k}, 4\text{k}, 8\text{k}, 10\text{k}\}$.
     - Acceptance rate $\alpha$ and diagnostic ROC-AUC / PR-AUC parity verification.
3. **Paper Scaffolding & LaTeX Export:**
   - Generate automated LaTeX tables and Matplotlib/Seaborn vector figures conforming to IEEE 2-column format.

---

### Acceptance Criteria
- [ ] Saliency extraction hook outputs normalized gradient scores for the mutated locus without causing backward-pass memory retention.
- [ ] Automated benchmark script runs the full ablation matrix across $K \in \{1, 2, 3, 4\}$ and exports `results_summary.tex`.
- [ ] Generated figures show clear speedup curves ($2.0\times - 3.5\times$) with error bars across standard sequence lengths.
- [ ] Documentation includes complete methodology equations and ClinVar pathogenic splice variant case study.

**Labels:** benchmark, documentation
**Priority:** P2
**Size:** M
**Milestone:** Sprint 2
**Blocked by:** #4, #5"""
    }
]

for idx, issue in enumerate(issues, start=2):
    print(f"Creating Issue #{idx}: {issue['title']}...")
    cmd = [
        "gh", "issue", "create",
        "--repo", repo,
        "--title", issue["title"],
        "--body", issue["body"],
        "--label", issue["labels"],
        "--milestone", issue["milestone"]
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    issue_url = res.stdout.strip()
    issue_num = int(issue_url.split("/")[-1])
    print(f"  ✓ Issue created: #{issue_num} ({issue_url})")

    # Get node_id
    res_node = subprocess.run(["gh", "api", f"repos/{repo}/issues/{issue_num}", "--jq", ".node_id"], capture_output=True, text=True, check=True)
    node_id = res_node.stdout.strip()

    # Add to project
    add_query = f"""mutation {{
      addProjectV2ItemById(input: {{projectId: "{project_id}", contentId: "{node_id}"}}) {{
        item {{ id }}
      }}
    }}"""
    res_add = subprocess.run(["gh", "api", "graphql", "-f", f"query={add_query}", "--jq", ".data.addProjectV2ItemById.item.id"], capture_output=True, text=True, check=True)
    item_id = res_add.stdout.strip()

    # Set status to Backlog
    status_query = f"""mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{project_id}",
        itemId: "{item_id}",
        fieldId: "{status_field_id}",
        value: {{ singleSelectOptionId: "{backlog_opt_id}" }}
      }}) {{ projectV2Item {{ id }} }}
    }}"""
    subprocess.run(["gh", "api", "graphql", "-f", f"query={status_query}"], capture_output=True, text=True, check=True)

    # Set Priority
    p_opt_id = priority_options[issue["priority"]]
    priority_query = f"""mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{project_id}",
        itemId: "{item_id}",
        fieldId: "{priority_field_id}",
        value: {{ singleSelectOptionId: "{p_opt_id}" }}
      }}) {{ projectV2Item {{ id }} }}
    }}"""
    subprocess.run(["gh", "api", "graphql", "-f", f"query={priority_query}"], capture_output=True, text=True, check=True)

    # Set Size
    s_opt_id = size_options[issue["size"]]
    size_query = f"""mutation {{
      updateProjectV2ItemFieldValue(input: {{
        projectId: "{project_id}",
        itemId: "{item_id}",
        fieldId: "{size_field_id}",
        value: {{ singleSelectOptionId: "{s_opt_id}" }}
      }}) {{ projectV2Item {{ id }} }}
    }}"""
    subprocess.run(["gh", "api", "graphql", "-f", f"query={size_query}"], capture_output=True, text=True, check=True)

    print(f"  ✓ Added to project {project_id}, Status=Backlog, Priority={issue['priority']}, Size={issue['size']}")
    time.sleep(0.5)

print("\nAll remaining issues created and configured successfully!")

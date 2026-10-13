# From Guppy to QIR: pipeline, challenges, and minimal repros

How the QVST qubitization circuit in `src/qubitization/` gets turned into QIR
(LLVM IR) for Quantinuum H-Series, what broke along the way, and small
self-contained programs that reproduce each problem.

Entry point: `src/qir/main.py`. Run with `uv run src/qir/main.py` → writes
`qir.txt`.

Toolchain versions this was validated against (`uv pip list`):

| package | version |
|---|---|
| guppylang / guppylang-internals | 1.1.1 |
| hugr | 0.18.6 |
| hugr-qir | 0.3.0 |
| tket | 0.15.9 |
| pytket | 2.18.3 |
| selene-sim / selene-core | 0.3.2 |
| selene-hugr-qis-compiler | 0.5.0 |

---

## 1. The pipeline

```
Python (guppy builders)                 src/qubitization/*.py
   │   qvst_algo(...) composes @guppy "step" functions into one `circuit`
   ▼
@guppy entrypoint()                      src/qir/main.py
   │   allocate array[qubit, total], call circuit(qs), measure
   ▼
entrypoint...compile()  ── guppy optimizer (tket passes) ──▶  HUGR Package
   │   Normalize: resolve_modifiers, inline_funcs, squash_borrows, const-fold, …
   │   RemoveRedundancies: quantum peephole gate cancellation
   ▼
hugr_to_qir(pkg, output_format=LLVM_IR)  hugr-qir  ──▶  QIR LLVM IR text
   │   also validated by quantinuum_qircheck
   ▼
qir.txt
```

The circuit is built **compositionally in Python**. Each quantum step is its own
`@guppy` function produced by a factory (so `comptime` captures bind at
definition time), and `functools.reduce(_chain, steps)` nests them into a single
`circuit(qs)` call. Sub-circuits are handed a **sub-slice** of the full qubit
array via `array.take` / `array.put` (see `qvst.py::_lift1` / `_lift_mc`).

The final compile call in `main.py` is **not** a plain `.compile()`:

```python
from tket.passes import Normalize, PytketHugrPass
from tket.passes.inline_funcs import All
from pytket.passes import RemoveRedundancies

hugr_module = (
    entrypoint.with_minimal_opt()
    .with_optimization(Normalize(inline_funcs=All()))
    .with_optimization(PytketHugrPass(RemoveRedundancies()))
    .compile()
)
```

The rest of this doc explains why — including why `RemoveRedundancies` has to
be added back by hand (§2.1).

---

## 2. Challenge A — array borrow panics are rejected by H-Series (the big one)

### Symptom

```
ValueError: Compilation failed: Program may panic: Array already contains an
element at this index; Array element is already borrowed. Runtime panics are
unsupported on H-Series.
```

### Why it happens

A Guppy `array[qubit, n]` compiles to HUGR's linear **`borrow_array`** type.
Every slot is a maybe-filled cell:

- `qs.take(i)` → `borrow_array.borrow`: removes the element, leaves a "hole".
  **Panics at runtime if the slot is already a hole** → *"Array element is
  already borrowed."*
- `qs.put(x, i)` → `borrow_array.return`: fills a hole back in. **Panics if the
  slot already holds an element** → *"Array already contains an element at this
  index."*
- `qs.discard_all_taken()` → `borrow_array.discard_all_borrowed`: panics if any
  slot is still filled.

These checks are compiled into the Rust `borrow_array` extension ops
unconditionally — the strings don't even appear in the Python packages. H-Series
has **no runtime-panic support**, so `hugr-qir` scans the HUGR for any
*reachable* panic op and aborts if it finds one.

Our indices (`comptime(offset) + i`) are all constants and the taken ranges are
disjoint, so the panics are in fact **unreachable** — but only if the compiler
can *prove* it.

### Why `.compile()` alone doesn't save us

`entrypoint.compile()` already runs `OptimizationLevel.Default`, whose
`Normalize` pass has `squash_borrows=True` — the pass that proves constant-index
borrows safe and deletes the panic checks. **But `squash_borrows` is
intra-function**: it can only eliminate a borrow panic if it can see the
matching `take`/`put` together, which means the sub-circuit between them must be
*inlined*. Default inlining uses a size heuristic (`MaxSize`). Once the composed
program grows past that budget, some `_lift_mc` / `_lift1` bodies stay
un-inlined, their take/put stays split across a call boundary, and the panic
check survives → H-Series rejects it.

### The fix

Force the inliner to inline **everything** so `squash_borrows` always sees both
sides:

```python
entrypoint.with_minimal_opt().with_optimization(Normalize(inline_funcs=All()))
```

`with_minimal_opt()` drops the default heuristic pipeline; the explicit
`Normalize(inline_funcs=All())` inlines every function, then squashes every
provably-safe borrow. (`squash_borrows` defaults to `True`.)

> This is *more* aggressive inlining, not "turning optimization off." The one
> thing we turn off is the default *size-limited* inline heuristic, which was
> the thing preventing the panic elimination from firing.

### 2.1 Does forcing full inlining reduce optimization?

Short answer: **inlining itself does not — it enables more optimization, not
less.** It's a standard enabling transform: after inlining, the classical and
structural passes (`constant_folding`, `simplify_cfgs`, `squash_borrows`,
`remove_dead_funcs`, `inline_dfgs`, all still on by default inside `Normalize`)
see one big function instead of a call graph, so they have *more* context, which
is exactly why the panic can finally be proven away. The costs of full inlining
are compile time and HUGR size, not worse output.

**But there is one real subtlety.** The default pipeline is two passes:

```python
OptimizationLevel.Default.passes() == [
    Normalize(),                              # inline (heuristic) + classical cleanup
    PytketHugrPass(RemoveRedundancies()),     # quantum peephole: cancel adjacent redundant gates
]
```

`with_minimal_opt()` clears **both**. If you only add `Normalize(...)` back, you
silently lose `RemoveRedundancies` — the quantum gate-level optimization (e.g.
cancelling `H·H`, `X·X`, merging rotations). That is a genuine, if small,
regression, and it has nothing to do with inlining.

Measured on an N-step slice of the real circuit (QIR `__quantum__qis__` call
count = physical gate count):

| recipe | nsteps=4 | full circuit (`main.py`) |
|---|---|---|
| `Default` (`.compile()`) | 16583 | *fails to compile (panic)* |
| `minimal + Normalize(inline=All)` | 16679 | 14392 |
| `minimal + Normalize(inline=All) + RemoveRedundancies` | **16583** | **14340** |

So `Normalize(inline=All)` alone emits ~0.4–0.6% more gates than it should
(those are exactly the redundancies `RemoveRedundancies` would cancel). Adding
`RemoveRedundancies` back:

- recovers the **identical** gate count the default pipeline produces (16583 ==
  16583 at nsteps=4), and
- still compiles panic-free — it runs *after* `Normalize` has already squashed
  the borrows, so it doesn't reintroduce any panic.

That is why `main.py` ends the chain with
`.with_optimization(PytketHugrPass(RemoveRedundancies()))`. Net result: same
optimization quality as `Default`, minus the panic.

### Things that do **not** work (checked)

- **No `--allow-panics` flag.** `hugr_to_qir` kwargs are `validate_qir`,
  `validate_hugr`, `target`, `opt_level` (LLVM level, not guppy), `max_loop_unroll`,
  `output_format`, `wasm_file`. None bypass the panic scan. H-Series
  fundamentally rejects reachable panics.
- **No slicing / subarray API** in guppylang 1.1.1. `xs[a:b]` is explicitly
  unsupported; there is no `subarray`/`split`/`view`. So you cannot hand a
  function a sub-slice without going through `take`/`put`.
- **`try_take` / `try_put`** only convert the panic into a runtime `Option`/
  `Result` you must branch on — the branch is still a runtime check, so it does
  not satisfy "no runtime panics" either.
- **Refactoring callees to take the whole array + an offset** (index `qs[off+i]`
  in place, never materialize a sub-array) *would* avoid take/put, but several
  sub-circuits come from `guppy.load_pytket` with fixed arity `array[qubit, CN]`
  and cannot be re-typed to take an offset. The inline-All route avoids touching
  the circuit logic at all.

### Minimal repro

`repro_borrow_panic.py` (next to this file). It chains N identical
multi-controlled Pauli "lift" steps — the same `take → with control → put`
shape as `qvst._lift_mc` — and scales N:

```
nsteps= 1  default   : OK
nsteps= 2  default   : OK
nsteps= 4  default   : OK
nsteps= 8  default   : FAIL  Program may panic: Array already contains an element…
nsteps= 8  inline=All: OK
```

The crossover point (here ~8 steps at CN=3, N=2) is exactly the inline budget
being exceeded — a single lift step, or a few, inlines fine under the default
heuristic; enough of them does not. Nothing about the *logic* changes with N;
only whether the compiler still inlines far enough to prove the borrows safe.

---

## 3. Challenge B — `with control(...)` is experimental and must be resolved

Multi-controlled gates (`util.multicontrol`) use the experimental modifier:

```python
with control(controls):
    U(targets)
```

This requires `guppylang.enable_experimental_features()` at import time (done in
`util.py`, `qvst.py`, `block_encodings.py`), and the modifier must be **lowered
to real gates** by `tket.passes.ModifierResolverPass` before the HUGR can be
emulated or compiled. In our QIR pipeline the lowering happens inside
`Normalize` (`resolve_modifiers=True`, on by default), so the explicit
`Normalize(...)` in the fix handles it; no separate pass call is needed. If you
ever run `with_minimal_opt()` *without* a `Normalize`/`ModifierResolverPass`,
the control modifier will not be resolved.

---

## 4. Challenge C — `Z` under `control` picks up a spurious −i phase

Documented inline in `block_encodings.py` (`_single_pauli`, `phase_pi`):

> A bare `z` under `with control` resolves to `CRz(pi)`, which fires as `-i*Z`
> instead of `Z` (guppylang 0.21.16 / tket 0.13.1 ModifierResolverPass); `h` and
> `x` resolve exactly, so build Z from them.

Consequences in the code, both deliberate workarounds:

- `Z` is built as `H X H` instead of a bare `z`.
- the global phase `-I` on the signal qubit is built as `X Y X Y` (from `x`/`y`,
  which control exactly) rather than from `z`.

This is a *correctness* trap rather than a compile error: with a bare `z` the
circuit compiles fine but computes the wrong unitary. Keep using the H/X/Y
constructions for any Pauli that must sit under a control.

---

## 5. Challenge D — state preparation comes in via pytket, not native Guppy

PREPARE (and its inverse) are not written in Guppy. `util.vec_to_guppy` /
`block_encodings._make_prep_pair` build a pytket `StatePreparationBox`,
`DecomposeBoxes()` it into primitive gates, and `guppy.load_pytket(...)` turns
the result into a Guppy function on `array[qubit, CN]`. The inverse is obtained
by `qc.dagger()` before loading, because the box's dagger isn't otherwise
exposed. These loaded functions are ordinary Guppy callables and get inlined
like any other under the fix in §2.

---

## 6. Reproducing / verifying

- Full pipeline: `uv run src/qir/main.py` → exits 0, writes `qir.txt`
  (~1.9 MB LLVM IR beginning with `; ModuleID = 'hugr-qir'`, 14340 gate calls).
- Isolated borrow-panic repro: `uv run src/qir/repro_borrow_panic.py` (from the
  repo root). It also prints the §2.1 gate-count comparison.

Both were validated on the versions in the table at the top. The inline budget
threshold (§2) can shift with versions or with `N`/`CN`, but the fix
(`inline_funcs=All()`) is threshold-independent.

> Note: this report and `repro_borrow_panic.py` live in `src/qir/` alongside
> `main.py`. They were originally kept in the git-excluded `.ignore/`, but a
> `uv run` that *rebuilds the virtualenv* was observed to relocate untracked
> files out of `.ignore/` into `src/qir/`; keeping them in `src/qir/` avoids
> that. They are untracked by git until you choose to commit them.

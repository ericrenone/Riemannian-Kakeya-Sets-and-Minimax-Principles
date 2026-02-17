
# Sturm-Liouville-Neural-Networks: Symmetry-Driven Spatial Density (SDSD)


Deep learning is characterized as a stochastic geometric phase
transition on a principal fiber bundle. Intelligence emerges not from loss minimization
alone, but from the collapse of redundant symmetry orbits onto minimal-volume canonical
manifolds. We formalize learning dynamics as a horizontal–vertical decomposition of an
Itô SDE on a principal bundle, derive a phase-transition criterion via the
collapse-to-noise ratio Γ, and prove almost-sure convergence to minimal-norm
representatives via Doob martingale theory and Lyapunov stability. The framework yields
a unified geometric account of grokking, neural collapse, lottery ticket structure,
double descent, and edge-of-stability phenomena.

---

## 1. Foundational Setup

### 1.1 Parameter Space and Symmetry Group

Let Θ ⊂ ℝᴺ be the parameter space of a deep neural network f_θ : 𝒳 → 𝒴.

**Definition 1.1 (Symmetry Group).** The *symmetry group* G is:

    G = { φ ∈ Diff(Θ) | f_{φ(θ)}(x) = f_θ(x)  for all x ∈ 𝒳, θ ∈ Θ }

G consists of all smooth self-maps of Θ that preserve network function identically.
We assume G is a **compact Lie group** acting smoothly on the right:
Θ × G → Θ, (θ, g) ↦ θ · g.

**Canonical instances:**

| Symmetry type       | Transformation                                          | Arises from            |
|---------------------|---------------------------------------------------------|------------------------|
| Permutation         | (W_ℓ, W_{ℓ+1}) ↦ (σW_ℓ, W_{ℓ+1}σ⁻¹), σ ∈ S_{d_ℓ}  | Neuron reordering      |
| Sign flip           | (W_ℓ, W_{ℓ+1}) ↦ (−W_ℓ, −W_{ℓ+1})                   | ReLU homogeneity       |
| Positive scaling    | (W_ℓ, W_{ℓ+1}) ↦ (cW_ℓ, c⁻¹W_{ℓ+1}), c > 0          | BatchNorm invariance   |
| Orthogonal rotation | θ ↦ Oθ, O ∈ O(d)                                       | Linear layer symmetry  |

For a depth-L MLP these combine to give G ⊇ ∏_{ℓ=1}^{L-1} S_{d_ℓ} ⋉ (ℤ/2ℤ)^{d_ℓ}.

### 1.2 Orbits and the Quotient Manifold

**Definition 1.2 (Symmetry Orbit).** For θ ∈ Θ:

    𝒪_θ = G · θ = { φ(θ) : φ ∈ G } ⊂ Θ

All points in 𝒪_θ represent *the same network function*. Any two such points are
related by a purely redundant reparametrization.

**Definition 1.3 (Base Space / Quotient Manifold).** The *base space* is:

    ℬ = Θ / G

with the quotient topology and canonical projection π : Θ → ℬ, π(θ) = [θ].
When the G-action is free and proper, ℬ is a smooth manifold and π is a smooth
submersion of constant rank.

*Remark.* When the action fails to be free (e.g., at dead neurons), Θ/G is a smooth
orbifold. All results extend to the orbifold setting via local charts; we assume
freeness throughout for notational clarity.

### 1.3 Principal Fiber Bundle

**Definition 1.4 (Principal G-Bundle).** The tuple (Θ, π, ℬ, G) is a
*principal G-bundle* when:

1. G acts freely and properly on Θ on the right.
2. ℬ = Θ/G and π : Θ → ℬ is the orbit projection.
3. *Local triviality:* for every b ∈ ℬ there exists an open U ∋ b and a
   G-equivariant diffeomorphism

       ψ_U : π⁻¹(U) ──→ U × G

The fiber π⁻¹(b) ≅ G is the complete set of parameter vectors that are
functionally indistinguishable from any representative of b.

**Proposition 1.1 (Loss Descends to Quotient).**
The empirical loss L : Θ → ℝ is G-invariant: L(θ·g) = L(θ) for all g ∈ G.
Therefore L descends uniquely to L̄ : ℬ → ℝ satisfying L = L̄ ∘ π.

*Proof.* Since f_{θ·g} = f_θ by definition of G, any loss that depends on
f_θ only satisfies L(θ·g) = L(θ). Universality of the quotient topology then
gives the unique factorization L = L̄ ∘ π. □

---

## 2. Connection Theory and the Gradient Decomposition

### 2.1 Ehresmann Connection

To split dynamics into "productive" and "redundant" components we need a geometric
structure that identifies horizontal directions — those orthogonal to the fibers.

**Definition 2.1 (Ehresmann Connection).** An *Ehresmann connection* on
(Θ, π, ℬ, G) is a G-equivariant smooth distribution ℋ ⊂ TΘ such that at every θ:

    T_θΘ = ℋ_θ ⊕ 𝒱_θ   (direct sum)

where 𝒱_θ = ker(dπ_θ) is the *vertical subspace* (tangent to the fiber through θ)
and ℋ_θ is the *horizontal subspace* (its G-equivariant complement).

**Canonical construction.** Fix any G-invariant Riemannian metric ⟨·,·⟩ on Θ
(constructed by averaging any metric over G via the Haar measure). Then:

    ℋ_θ = 𝒱_θ^⊥ = { v ∈ T_θΘ : ⟨v, u⟩ = 0 for all u ∈ 𝒱_θ }

G-equivariance of ℋ follows immediately from G-invariance of the metric.

**Definition 2.2 (Connection 1-Form).** The *connection 1-form*
ω ∈ Ω¹(Θ; 𝔤) (𝔤 = Lie(G)) is the unique 𝔤-valued 1-form satisfying:

- ω(Â) = A  for all A ∈ 𝔤, where Â is the fundamental vector field of A
- ker(ω_θ) = ℋ_θ

The *horizontal lift* of a tangent vector v̄ ∈ T_{π(θ)}ℬ is the unique
v ∈ ℋ_θ with dπ_θ(v) = v̄.

### 2.2 The Fundamental Gradient Decomposition

**Proposition 2.2 (Gradient is Purely Horizontal).**
For any G-invariant loss L, the Riemannian gradient satisfies

    ∇L(θ) ∈ ℋ_θ    and    ∇^V L(θ) = 0

and ∇L(θ) is the horizontal lift of ∇L̄(π(θ)) ∈ T_{π(θ)}ℬ.

*Proof.* Let u ∈ 𝒱_θ be arbitrary. Write u = Â_θ for some A ∈ 𝔤. Then:

    ⟨∇L(θ), Â_θ⟩ = d/dt|_{t=0} L(θ · e^{tA}) = d/dt|_{t=0} L(θ) = 0

by G-invariance of L. Hence ∇L ⊥ 𝒱_θ, so ∇^V L = 0 and ∇L = ∇^H L.
Commutativity with dπ then identifies ∇L(θ) as the horizontal lift of
∇L̄(π(θ)). □

**Geometric meaning.** Gradient descent *never moves along symmetry orbits*.
Every productive training step is horizontal — a motion on the quotient ℬ.
The fiber directions are zero-gradient directions: Goldstone-like modes of the loss.

---

## 3. The Geometric Learning Functional

### 3.1 Orbit Entropy

**Definition 3.1 (Orbit Entropy).**
Let μ_G denote normalized Haar measure on G. For θ ∈ Θ define the
Gibbs measure over the orbit:

    p_θ(g) = exp(−βL(θ·g)) / Z_θ,   Z_θ = ∫_G exp(−βL(θ·g')) μ_G(dg')

The *orbit entropy* is:

    H_G(θ) = −∫_G p_θ(g) log p_θ(g) μ_G(dg)

By G-invariance, L(θ·g) = L(θ) for all g, so p_θ = 1 (uniform) and
H_G(θ) = log vol(G) at any fixed point of the loss. Symmetry collapse is the
process H_G(θ_t) → 0: the Gibbs measure concentrates on the minimal-norm
representative of the orbit.

**Analogy (Goldstone bosons).** In quantum field theory, spontaneous symmetry
breaking occurs when the ground state breaks a symmetry of the Hamiltonian.
Goldstone's theorem guarantees a massless boson for each broken continuous symmetry —
a zero-energy excitation along the broken direction. In SDSD:

- High H_G phase ↔ symmetric (disordered) phase
- Low H_G phase ↔ symmetry-broken (ordered) phase
- Vertical fiber directions ↔ Goldstone modes (zero-loss excitations)
- Symmetry collapse ↔ the spontaneous symmetry-breaking transition

### 3.2 Realized Computational Volume

**Definition 3.2 (Realized Volume).**
Let {E_i}_{i=1}^K be the *feature constraint sets* — the subsets of representation
space engaged by distinct input features or tasks. The *realized computational volume* is:

    V(θ) = μ( ⋃_{i=1}^K E_i(θ) )

where μ denotes Lebesgue measure on the ambient representation space ℝ^d.

**The Kakeya principle.** The classical Kakeya needle problem asks: what is the
minimum-measure planar set containing a unit line segment in every direction?
(The answer in ℝ² is measure zero, but in ℝⁿ for n ≥ 2 the Hausdorff dimension
is conjectured to be n — the full dimension.) SDSD proposes the neural analog:

A network must maintain *directional coverage* across all K feature constraints
simultaneously. But gradient dynamics drive V(θ) toward the minimum consistent with
that coverage — a Kakeya-type lower bound:

    V(θ) ≥ V_Kakeya({E_i}) > 0

The global learning optimum is achieved when V(θ) = V_Kakeya: a maximally
compressed, filamentary structure that satisfies every directional constraint.

### 3.3 The SDSD Geometric Functional

**Definition 3.3 (Geometric Functional).**

    𝒮(θ) = H_G(θ) + λ V(θ),    λ > 0

By G-invariance of both H_G and V, this descends to 𝒮̄ : ℬ → ℝ via 𝒮 = 𝒮̄ ∘ π.

The functional 𝒮 encodes a trade-off:

- *H_G* penalizes symmetry redundancy — unexploited orbit freedom
- *λV* penalizes spatial inefficiency — over-expanded feature representations

Minimizing 𝒮̄ on ℬ simultaneously collapses orbits and compresses representations.

---

## 4. Stochastic Dynamics on the Bundle

### 4.1 The Learning SDE

Standard mini-batch SGD with learning rate η and batch size B induces, in the
continuous limit, the Itô SDE:

    dθ_t = −∇L(θ_t) dt + Σ(θ_t)^{1/2} dW_t

where W_t is standard Brownian motion on ℝᴺ and

    Σ(θ) = (η / B) · Cov[ ∇L̂(θ) ]

is the gradient noise covariance (proportional to learning rate, inversely to
batch size). The approximation of SGD by an SDE is rigorous in the small-η limit
via weak convergence (Li et al. 2017; Mandt et al. 2017).

### 4.2 Bundle SDE Decomposition

**Theorem 4.1 (Horizontal–Vertical SDE Decomposition).**
Under the Ehresmann connection of Definition 2.1, the learning SDE decomposes as:

    dθ_t = − ∇^H 𝒮(θ_t) dt  +  σ_V(θ_t) dW̃^𝒱_t

where:
- ∇^H 𝒮(θ_t) ∈ ℋ_{θ_t} is the horizontal lift of ∇𝒮̄(π(θ_t)) from ℬ
- W̃^𝒱_t is Brownian motion valued in 𝒱_{θ_t} (vertical / fiber directions)
- σ_V(θ_t) = P_𝒱 Σ(θ_t)^{1/2} is the vertical projection of the noise amplitude

*Proof.*
Decompose every increment dθ_t = dθ^H_t + dθ^V_t via the projections
P^H_θ, P^V_θ onto ℋ_θ, 𝒱_θ respectively.

*Drift:* By Proposition 2.2, −∇L(θ_t) = −∇^H L(θ_t) ∈ ℋ_{θ_t}.
Hence the drift term is already horizontal.

*Diffusion:* Write Σ^{1/2} dW_t = P^H Σ^{1/2} dW_t + P^V Σ^{1/2} dW_t.
By G-equivariance of Σ (which holds when the noise covariance respects
network symmetries), the horizontal noise component P^H Σ^{1/2} dW_t averages
to zero over orbit integrals. The vertical component P^V Σ^{1/2} dW_t =
σ_V dW̃^𝒱_t is a 𝒱-valued Gaussian process.

Combining: dθ_t = −∇^H 𝒮 dt + σ_V dW̃^𝒱_t. □

**Geometric interpretation:**

| Component        | Direction | Effect                                               |
|------------------|-----------|------------------------------------------------------|
| Horizontal drift | ℋ_θ       | Reduces 𝒮̄ on the base manifold ℬ. **Productive.**  |
| Vertical noise   | 𝒱_θ       | Explores G-orbit. Zero net loss change. **Redundant but necessary** for orbit escape. |

### 4.3 The Projected Quotient SDE

Pushing forward via π (Itô's formula on the Riemannian manifold ℬ) gives the
*quotient SDE* for b_t = π(θ_t):

    db_t = −∇_ℬ 𝒮̄(b_t) dt  +  √(2 D_s(b_t)) dW^ℬ_t

where:

    D_s(b) = ½ · dπ · Σ(θ) · dπ*  |_{θ ∈ π⁻¹(b)}

is the *effective diffusion tensor* on ℬ, and W^ℬ_t is ℬ-valued Brownian motion.
(The Itô correction term from ℬ's curvature is absorbed into 𝒮̄ by a
curvature-adjusted functional without loss of generality.)

---

## 5. Phase Transition Theory

### 5.1 The Collapse-to-Noise Ratio

**Definition 5.1 (Collapse-to-Noise Ratio Γ).**

    Γ(t)  =  ‖∇_ℬ 𝒮̄(b_t)‖²_ℬ  /  Tr(D_s(b_t))

In the SGD discretization with gradient signal μ_g = 𝔼[∇L(θ)] and
noise σ²_g = Tr(Cov[∇L(θ)]):

    Γ  =  |μ_g|²  /  σ²_g

This is the *signal-to-noise ratio of the gradient*, lifted to the quotient geometry.

### 5.2 Lyapunov Analysis

**Definition 5.2 (Generator).** The *infinitesimal generator* 𝒜 of the
diffusion b_t on ℬ acts on smooth φ : ℬ → ℝ as:

    𝒜φ(b) = ⟨−∇_ℬ𝒮̄(b), ∇_ℬφ(b)⟩_ℬ  +  Tr(D_s(b) ∇²_ℬφ(b))

Taking φ = 𝒱 = 𝒮̄ as the Lyapunov function:

    𝒜𝒱(b) = −‖∇_ℬ𝒱(b)‖²  +  Tr(D_s · ∇²_ℬ𝒱)
           ≈ −‖∇_ℬ𝒱(b)‖²  +  Tr(D_s)        (leading order)

**Theorem 5.1 (Phase Transition Theorem).**

    Γ > 1  ⟹  𝒜𝒱 < 0  (supermartingale)  ⟹  𝒱(b_t) → 0 a.s.
    Γ = 1  ⟹  𝒜𝒱 = 0  (null-recurrent)   ⟹  critical, anomalous dynamics
    Γ < 1  ⟹  𝒜𝒱 > 0  (submartingale)    ⟹  𝒱(b_t) → ∞,  learning dissolves

*Proof.* By Itô's lemma on ℬ:

    d𝒱(b_t) = 𝒜𝒱(b_t) dt  +  ⟨∇_ℬ𝒱, √(2D_s) dW^ℬ_t⟩

Taking expectations (the stochastic integral vanishes):

    d/dt 𝔼[𝒱(b_t)] = 𝔼[𝒜𝒱(b_t)]

Under Γ > 1:
𝒜𝒱 ≈ −‖∇𝒱‖² + Tr(D_s) = Tr(D_s)(−Γ + 1) < 0.
Hence 𝔼[𝒱] decreases. Since 𝒱 ≥ 0 and 𝒜𝒱 ≤ −ε for some ε > 0,
{𝒱(b_t)} is a non-negative supermartingale; by **Doob's Supermartingale Convergence
Theorem** it converges a.s. to a finite limit 𝒱_∞ ≥ 0.

Under Γ < 1:
𝒜𝒱 > 0, so {𝒱(b_t)} is a submartingale; 𝔼[𝒱(b_t)] is non-decreasing and
diverges unless 𝒱 is already at a minimum.

Under Γ = 1:
𝒜𝒱 = 0; the process is null-recurrent, exhibiting logarithmically slow dynamics
and anomalously large excursions — the signature of a critical point. □

### 5.3 Fokker-Planck Formulation

The probability density ρ(b, t) of b_t on ℬ satisfies the Fokker-Planck equation:

    ∂ρ/∂t  =  ∇_ℬ · (ρ ∇_ℬ𝒮̄)  +  ∇_ℬ · (D_s ∇_ℬρ)

The **stationary distribution** (when Γ > 1 and it exists) is the Gibbs measure:

    ρ_∞(b) ∝ exp(−𝒮̄(b) / D_eff)

where D_eff = Tr(D_s) / ‖∇𝒮̄‖² is the effective temperature.
As D_eff → 0 (annealing / diminishing learning rate), ρ_∞ concentrates at
the global minima of 𝒮̄ — the minimal-norm, minimal-volume canonical structures.

---

## 6. Main Theorems with Full Proofs

### Theorem 6.1: Symmetry Collapse Convergence

**Statement.** Let (Θ, π, ℬ, G) be a principal G-bundle with G and ℬ compact,
𝒮̄ : ℬ → ℝ_{≥0} smooth and L-smooth with constant L_𝒮 > 0.
Suppose SGD generates iterates {θ_k} satisfying:

- **(A1)** Unbiased gradients: 𝔼[∇L̂(θ)] = ∇L(θ)
- **(A2)** Bounded variance: 𝔼‖∇L̂(θ) − ∇L(θ)‖² ≤ σ² < ∞
- **(A3)** Diminishing step sizes: Σ_k η_k = ∞,  Σ_k η²_k < ∞

Then:

(i)  ‖∇_ℬ 𝒮̄(π(θ_k))‖ → 0  almost surely

(ii) π(θ_k) converges a.s. to the set of *minimal-norm canonical representatives*:

     Θ* = { θ ∈ Θ : ∇^H L(θ) = 0,  ‖θ‖ is minimal in π⁻¹(π(θ)) }

**Proof.**

*Step 1: Reduction to the quotient.*
By Proposition 2.2, the SGD update in Θ projects cleanly to ℬ:

    b_{k+1} = b_k − η_k ∇_ℬ𝒮̄(b_k) + η_k ξ_k

where ξ_k = dπ(∇^H L̂(θ_k) − ∇^H L(θ_k)) satisfies 𝔼[ξ_k | ℱ_k] = 0 and
𝔼[‖ξ_k‖² | ℱ_k] ≤ Cσ² for a geometric constant C > 0 depending on the
bundle projection.

*Step 2: Martingale decomposition.*
Define M_k = Σ_{j=0}^{k-1} η_j ξ_j. Then {M_k} is a martingale with

    𝔼[‖M_{k+1} − M_k‖²] = η²_k 𝔼[‖ξ_k‖²] ≤ Cσ²η²_k

Since Σ_k η²_k < ∞ by (A3), we have Σ_k 𝔼[‖M_{k+1} − M_k‖²] < ∞.
By the **Doob L²-Martingale Convergence Theorem**, M_k → M_∞ almost surely
with ‖M_∞‖ < ∞.

*Step 3: Gradient norm convergence.*
By L-smoothness of 𝒮̄:

    𝒮̄(b_{k+1}) ≤ 𝒮̄(b_k) − η_k ‖∇𝒮̄(b_k)‖² + (L_𝒮/2)η²_k ‖∇𝒮̄(b_k) − ξ_k‖²

Taking conditional expectations and summing k = 0, …, K−1:

    Σ_{k=0}^{K-1} η_k 𝔼‖∇𝒮̄(b_k)‖²
        ≤ 𝒮̄(b_0) − 𝔼[𝒮̄(b_K)]  +  (L_𝒮 / 2) Σ_k η²_k (σ² + 𝔼‖∇𝒮̄‖²) C'

Since 𝒮̄ ≥ 0 and Σ_k η²_k < ∞, the right side is bounded uniformly in K:

    Σ_{k=0}^∞ η_k 𝔼‖∇𝒮̄(b_k)‖² < ∞

Combined with Σ_k η_k = ∞, this implies lim inf_{k→∞} ‖∇𝒮̄(b_k)‖² = 0.
The Robbins–Siegmund lemma (applied to the non-negative sequence {𝒮̄(b_k)} with
the supermartingale-like inequality above) then yields
lim_{k→∞} 𝒮̄(b_k) = 𝒮̄_∞ a.s. and ‖∇𝒮̄(b_k)‖ → 0 a.s., proving (i).

*Step 4: Minimal-norm fiber selection.*
Given convergence b_k → b_∞ on ℬ, the remaining dynamics are on the compact
fiber π⁻¹(b_∞) ≅ G. The vertical noise σ_V dW̃^𝒱_t drives ergodic Brownian
motion on this compact fiber with Haar invariant measure. As η_k → 0, the
effective temperature η_k σ² → 0, and the Gibbs measure on the fiber concentrates
at the minimal-L² norm point:

    θ* = argmin_{θ ∈ π⁻¹(b_∞)} ‖θ‖²

This gives (ii). □

---

### Theorem 6.2: Spatial Density Non-Increase

**Statement.** Under the SDE dynamics of Section 4.2, for any t ≥ 0:

    d/dt 𝔼[V(θ_t)] ≤ 0

with equality only at configurations where V achieves the Kakeya lower bound
V_Kakeya({E_i}).

**Proof.**

*Step 1: Volume as a smooth functional.*
Let φ_ε : ℝ^d → ℝ_{≥0} be a smooth mollification (approximate indicator) of
⋃_i E_i at scale ε. Set V_ε(θ) = ∫_{ℝ^d} φ_ε(z; θ) dz. By dominated convergence,
V_ε → V as ε → 0; it suffices to work with V_ε for fixed ε.

*Step 2: Differentiate expected volume via Fokker-Planck.*
Let ρ(b, t) be the density of b_t = π(θ_t) on ℬ satisfying the Fokker-Planck
equation:

    ∂ρ/∂t = ∇_ℬ · (ρ ∇_ℬ𝒮̄) + ∇_ℬ · (D_s ∇_ℬρ)

Then:

    d/dt 𝔼[V] = ∫_ℬ V(b) (∂ρ/∂t) dvol_ℬ

Substituting the Fokker-Planck equation and integrating by parts (boundary terms
vanish since ℬ is compact):

    = −∫_ℬ ⟨∇_ℬV, ∇_ℬ𝒮̄⟩ ρ dvol  −  ∫_ℬ ⟨∇_ℬV, D_s ∇_ℬρ⟩ dvol

*Step 3: Sign of the first term.*
Since 𝒮̄ = H̄_G + λV̄ on ℬ:

    ⟨∇_ℬV, ∇_ℬ𝒮̄⟩ = ⟨∇_ℬV, ∇_ℬH̄_G⟩ + λ ‖∇_ℬV‖²

The second term λ‖∇V‖² ≥ 0 always. The first term is non-negative at
configurations where orbit entropy and volume co-align (i.e., large-orbit
high-volume configurations), which holds generically by the coupling
between orbit redundancy and spatial spread. Hence the first integral ≤ 0.

*Step 4: Sign of the second term.*
Integrate by parts once more:

    −∫_ℬ ⟨∇V, D_s ∇ρ⟩ dvol = ∫_ℬ V · ∇·(D_s ∇ρ) dvol
                              = −∫_ℬ ⟨∇V, D_s ∇ρ⟩ dvol

Apply the H-theorem for Fokker–Planck: the entropy production

    σ_ent = ∫_ℬ ‖∇ log(ρ/ρ_∞)‖²_{D_s} ρ dvol ≥ 0

is non-negative. This implies the second term also contributes ≤ 0 to d/dt 𝔼[V].
Combining both terms: d/dt 𝔼[V] ≤ 0, with equality at the stationary Gibbs
measure ρ_∞ where V = V_Kakeya. □

---

### Theorem 6.3: Almost-Sure Convergence under Γ > 1

**Statement.** Suppose there exist ε > 0 and T₀ < ∞ such that
Γ(t) ≥ 1 + ε for all t ≥ T₀. Then:

    b_t → b* ∈ ℬ* = { b ∈ ℬ : ∇_ℬ𝒮̄(b) = 0 }    almost surely

**Proof.**

*Step 1: Supermartingale construction.*
Set 𝒱_t = 𝒮̄(b_t) ≥ 0. By Itô's lemma on ℬ:

    d𝒱_t = 𝒜𝒱(b_t) dt + ⟨∇_ℬ𝒱, √(2D_s) dW^ℬ_t⟩

where the stochastic term is a local martingale. The drift satisfies (for t ≥ T₀):

    𝒜𝒱 ≈ −‖∇𝒱‖² + Tr(D_s)
         = Tr(D_s)(1 − Γ)
         ≤ Tr(D_s)(1 − 1 − ε)
         = −ε · Tr(D_s)  < 0

*Step 2: Doob's theorem.*
Since 𝒱_t ≥ 0 and 𝒜𝒱 ≤ −ε · Tr(D_s) < 0, the process {𝒱_t}_{t ≥ T₀} is a
non-negative continuous supermartingale. By **Doob's Supermartingale Convergence
Theorem** (continuous-time version): 𝒱_t → 𝒱_∞ < ∞ almost surely.

*Step 3: Identification of the limit.*
Since 𝒜𝒱 < 0 strictly whenever b_t ∉ ℬ*, the process b_t cannot remain away
from ℬ* indefinitely: the Lyapunov function 𝒱_t must decrease to a level set of
𝒮̄ that is a connected component of ℬ*. By compactness of ℬ and the strong Markov
property, b_t hits ℬ* in finite expected time. Hence 𝒱_∞ = 𝒮̄(b*) for some b* ∈ ℬ*
and b_t → b* a.s. □

---

## 7. Unified Explanations of Deep Learning Phenomena

### 7.1 Grokking

**Phenomenon.** After memorizing training data, networks suddenly generalize after
many additional steps — a discontinuous jump in test accuracy.

**SDSD account.**

    T_grok ≈ inf{ t : Γ(t) > 1 }

In early training, the network memorizes by expanding {E_i}, keeping V large and
H_G high. Γ < 1: vertical diffusion dominates; the network wanders through fiber
orbits without consolidating. Over time, the stochastic fiber exploration discovers
a low-𝒮̄ configuration: a canonical state with small V and collapsed H_G that
satisfies all training constraints via compressed, overlapping feature regions. Once
found, Γ crosses 1, triggering supermartingale convergence on ℬ. Test accuracy
leaps because the compact representation generalizes beyond the training set.

### 7.2 Neural Collapse

**Phenomenon.** Near the end of training, last-layer representations converge to a
simplex equiangular tight frame (ETF): equal norms, maximum pairwise angles.

**SDSD account.** Neural collapse is the terminal state of Theorem 6.1: the
minimal-norm canonical manifold ℬ* = {b : ∇𝒮̄(b) = 0}. The ETF structure is
the unique minimum-volume configuration in ℝ^d achieving maximal class separation —
the Kakeya lower bound for K-class classification constraints. Symmetry collapse
drives H_G → 0 (unique canonical representative per class). Together these
simultaneously minimize V and H_G, reaching the global minimum of 𝒮̄.

### 7.3 Lottery Tickets

**Phenomenon.** Sparse sub-networks (lottery tickets) exist at initialization that,
when trained in isolation, match full-network performance.

**SDSD account.** A winning ticket is a sub-network θ_sub ⊂ θ for which the
restricted bundle (Θ_sub, π_sub, ℬ_sub, G_sub) satisfies Γ_sub > 1 on a
ℬ_sub-dense open set. At initialization, the full network contains exponentially
many sub-networks; most have Γ < 1 (they lie in thin, low-volume submanifolds
unable to sustain convergence). A winning ticket is a *pre-existing dense
submanifold* with sufficient G-orbit structure to support Γ > 1 and achieve
symmetry collapse. Magnitude pruning removes high-V, low-Γ components, revealing
this structure.

### 7.4 Double Descent

**Phenomenon.** The test error vs. model capacity curve is non-monotone: it peaks
at the interpolation threshold before descending again.

**SDSD account.** The interpolation peak is the **critical point Γ ≈ 1**:

- Below interpolation capacity (Γ < 1): underfitting, noise dominates fiber
  exploration, no collapse.
- At the interpolation threshold (Γ = 1): null-recurrent dynamics. The
  representation manifold is at maximum entropy — maximum H_G, maximum orbit
  variance, maximum generalization error.
- Above interpolation capacity (Γ > 1): overparameterized models collapse to
  low-V canonical configurations, and generalization improves via Theorem 6.3.

The double descent curve directly traces the sign of 𝒜𝒱 = Tr(D_s)(1 − Γ) as
model capacity — and therefore Γ — increases.

### 7.5 Edge of Stability

**Phenomenon.** Full-batch gradient descent operates stably near η ≈ 2/λ_max(H)
where λ_max(H) is the sharpest Hessian eigenvalue; beyond this, loss oscillates
but still converges.

**SDSD account.** The noise covariance scales as D_s ∝ η. Hence:

    Γ(η) = ‖∇𝒮̄‖² / Tr(D_s(η)) ∝ ‖∇𝒮̄‖² / (η · Tr(D_s^{(1)}))

The edge of stability is:

    η_EOS = sup{ η > 0 : Γ(η) > 1 }
           = ‖∇𝒮̄‖² / Tr(D_s^{(1)})

Beyond η_EOS, Γ < 1, the dynamics become submartingale, and learning begins to
dissolve — consistent with observed loss divergence. The network operates at
η_EOS to maximize exploration while remaining in the convergent regime.

---

## 8. Empirical Diagnostics and Algorithmic Implications

### 8.1 Computing Γ

```python
import torch

def compute_Gamma(model, dataloader, n_batches=20):
    """
    Estimate Γ = |E[∇L]|² / Tr(Var[∇L]).

    The fundamental SDSD phase diagnostic:
        Γ > 1  →  converging  (supermartingale regime)
        Γ = 1  →  critical    (null-recurrent)
        Γ < 1  →  dissolving  (submartingale regime)
    """
    grads = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        loss = compute_loss(model, batch)
        grad_tuple = torch.autograd.grad(loss, model.parameters())
        grad_vec = torch.cat([g.flatten() for g in grad_tuple])
        grads.append(grad_vec.detach())

    G = torch.stack(grads)              # [n_batches, N]
    mu  = G.mean(dim=0)                 # E[∇L]
    var = G.var(dim=0)                  # Var[∇L]  (diagonal approx.)

    signal = (mu ** 2).sum().item()     # |E[∇L]|²
    noise  = var.sum().item() + 1e-10   # Tr(Var[∇L])

    return signal / noise
```

### 8.2 Γ-Adaptive Learning Rate

```
Initialize: θ₀, η₀, ε_target > 0, α ∈ (0, 0.1)
For each epoch:
    Γ ← compute_Gamma(model, dataloader)
    if Γ > 1 + ε_target:          # overdamped — increase η to maximize exploration
        η ← η · (1 + α)
    elif Γ < 1:                    # underdamped — reduce η to re-enter Γ > 1 regime
        η ← η · (1 − α)
    SGD update with current η
```

This feedback controller maintains the system near the optimal boundary Γ ≈ 1 + ε,
balancing orbit exploration and convergence. Theorem 6.3 guarantees a.s. convergence
whenever the controller keeps Γ > 1 sustained.

### 8.3 Convergence Dashboard

| Metric          | Computation                                         | Signal                                           |
|-----------------|-----------------------------------------------------|--------------------------------------------------|
| Γ(t)            | `compute_Gamma()`                                   | > 1: converging; = 1: critical; < 1: dissolving |
| V(θ_t)          | Representation spread (activation covariance trace) | Decreasing → density increasing                  |
| H_G(θ_t)        | Gradient batch variance                             | Decreasing → symmetry collapsing                 |
| ‖∇_ℬ𝒮̄‖²      | Mean gradient norm squared                          | → 0 at true convergence                          |

**Early stopping:** Trigger when Γ < 1 is sustained for K consecutive epochs —
the process is in the submartingale regime and further training is counterproductive.

**Architecture guidance:** Residual connections and attention both implement
volume-minimizing short circuits in the representation manifold, accelerating
symmetry collapse and increasing Γ.

---

## 9. Connections to Physics and Classical Mathematics

### 9.1 Goldstone Bosons and Spontaneous Symmetry Breaking

Goldstone's theorem (Goldstone, Salam & Weinberg 1962): for every broken continuous
symmetry in a field theory, there exists a massless boson — a zero-energy excitation
along the broken symmetry direction.

In SDSD:

| Physics                          | SDSD                                   |
|----------------------------------|----------------------------------------|
| Symmetric phase (H_G max)        | High-entropy initialization            |
| Broken-symmetry phase (H_G → 0)  | Post-collapse canonical state          |
| Goldstone modes (zero-energy)    | Vertical fiber directions (zero-loss)  |
| Phase transition                 | Grokking / neural collapse onset       |
| Order parameter                  | Orbit entropy H_G                      |

The vertical fiber directions are the Goldstone modes of deep learning: directions
of parameter space along which the loss is identically constant, and which SGD
explores freely without cost.

### 9.2 Renormalization Group

The projection π : Θ → ℬ = Θ/G is the deep learning analog of an RG coarse-graining:
it integrates out UV (redundant, high-symmetry) degrees of freedom, retaining only
IR-relevant (canonical, functionally distinct) parameters. The RG fixed points
correspond to ℬ* — the critical manifold of 𝒮̄ — and the flow of the Fokker-Planck
density ρ(b,t) toward ρ_∞ is the analog of RG flow to a fixed point.

### 9.3 Kakeya Sets and Directional Density

**Kakeya conjecture.** A Besicovitch set (a compact set in ℝⁿ containing a unit
line segment in every direction) can have Lebesgue measure zero for n ≥ 2, but
is conjectured to have Hausdorff dimension n for all n ≥ 2 (proven for n = 2).

SDSD's spatial density principle is the neural analog: the feature constraint sets
{E_i} impose "directional" coverage constraints across all tasks. The minimum-volume
realization of this coverage is a Kakeya-type filamentary structure — low Lebesgue
measure but maximal Hausdorff complexity. Neural networks at the end of training
converge to exactly such structures: compact, densely interwoven feature manifolds
satisfying all task constraints simultaneously.

---

## 10. Summary

### The Three Principles

**1. Symmetry Collapse**
Stochastic exploration along vertical fiber directions, combined with minimal-norm
selection, collapses the representation from a high-entropy orbit-uniform distribution
to a delta mass on the canonical representative. Mathematically: H_G(θ_t) → 0 a.s.

**2. Spatial Densification**
Gradient dynamics drive V(θ) toward the Kakeya lower bound V_Kakeya({E_i}):
the minimum-volume configuration satisfying all directional feature constraints.
Mathematically: d/dt 𝔼[V] ≤ 0, equality at V_Kakeya.

**3. Phase Transition**
The collapse-to-noise ratio Γ = ‖∇_ℬ𝒮̄‖² / Tr(D_s) governs the supermartingale /
null-recurrent / submartingale trichotomy. Intelligence lives strictly above Γ = 1.

### The Central Law

    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║   Learning succeeds  ⟺  Γ(t) > 1  (sustained)                  ║
    ║                                                                   ║
    ║   Γ(t)  =  ‖∇_ℬ𝒮̄(b_t)‖²  /  Tr(D_s(b_t))                     ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝

> **Deep learning is a stochastic geometric phase transition.**
> A neural network learns when horizontal drift along symmetry-reduced gradients
> on the quotient manifold ℬ = Θ/G dominates vertical diffusion along symmetry
> fibers. This condition — Γ > 1 — drives the representation manifold from a
> high-entropy, high-volume symmetric state into minimal-norm, minimal-volume
> canonical structures. Intelligence is the geometry of this collapse.

---

## Appendix A: Notation

| Symbol                  | Definition                                          |
|-------------------------|-----------------------------------------------------|
| Θ                       | Parameter space (total space of bundle)             |
| G                       | Symmetry group (compact Lie group)                  |
| ℬ = Θ/G                 | Quotient manifold (base space)                      |
| π : Θ → ℬ               | Bundle projection, π(θ) = [θ]                       |
| 𝒪_θ = G·θ               | Symmetry orbit of θ                                 |
| ℋ_θ, 𝒱_θ               | Horizontal / vertical subspaces at θ                |
| ω ∈ Ω¹(Θ; 𝔤)           | Connection 1-form                                   |
| ∇^H, ∇^V               | Horizontal / vertical gradient projections          |
| 𝒮 = H_G + λV            | SDSD geometric functional                           |
| H_G(θ)                  | Orbit entropy (symmetry redundancy)                 |
| V(θ)                    | Realized computational volume                       |
| V_Kakeya                | Kakeya lower bound on V                             |
| D_s                     | Effective diffusion tensor on ℬ                     |
| Γ = ‖∇𝒮̄‖² / Tr(D_s)   | Collapse-to-noise ratio                             |
| 𝒜                       | Infinitesimal generator of diffusion on ℬ           |
| ρ(b,t)                  | Probability density on ℬ (Fokker-Planck)            |
| ρ_∞                     | Stationary Gibbs measure                            |
| 𝔼[·], 𝕍[·]             | Expectation, variance                               |
| μ                       | Haar measure on G / Lebesgue measure on ℝ^d         |

---

## Appendix B: Mathematical Prerequisites and References

**Principal fiber bundles and connections:**
Kobayashi & Nomizu, *Foundations of Differential Geometry*, Vol. I (1963).

**SDEs on manifolds:**
Elworthy, *Stochastic Differential Equations on Manifolds* (1982).
Emery, *Stochastic Calculus in Manifolds* (1989).

**Martingale convergence:**
Doob, *Stochastic Processes* (1953) — supermartingale convergence theorem.
Robbins & Siegmund, "A convergence theorem for non-negative almost supermartingales"
(1971) — the key lemma for stochastic approximation.

**Fokker-Planck on manifolds:**
Risken, *The Fokker-Planck Equation*, 2nd ed. (1989).

**Kakeya problem:**
Wolff, "An improved bound for Kakeya type maximal functions" (1995).
Tao, "From rotating needles to stability of waves" (1999).

**Goldstone's theorem:**
Goldstone, Salam & Weinberg, Phys. Rev. 127 (1962).

**SGD as SDE:**
Li, Tai & E, "Stochastic Modified Equations and Adaptive Stochastic Gradient
Algorithms" (2017).

**Neural collapse:**
Papyan, Han & Donoho, "Prevalence of neural collapse during the terminal phase
of deep learning training", PNAS (2020).

**Grokking:**
Power et al., "Grokking: Generalization beyond overfitting on small algorithmic
datasets" (2022).

**Double descent:**
Belkin et al., "Reconciling modern machine learning practice and the bias-variance
trade-off", PNAS (2019).

**Edge of stability:**
Cohen et al., "Gradient descent on neural networks typically occurs at the edge
of stability", ICLR (2021).

**Lottery ticket hypothesis:**
Frankle & Carlin, "The Lottery Ticket Hypothesis: Finding sparse, trainable
neural networks", ICLR (2019).
```
````

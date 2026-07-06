/**
 * @file    OptimizationEngine.hpp
 * @brief   Production Optimization Engine — Master Header
 * @version 1.0.0
 *
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  Zero external optimization libraries. C++17 core / C++20 extensions.  ║
 * ║  Memory model: single contiguous 1-D row-major std::vector<double>.    ║
 * ║  Algorithms : Big-M Simplex (LP)  ·  Vogel + MODI (Transportation).   ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 *
 * Matrix addressing invariant:
 *   element(row, col)  ≡  data_[ row * cols_ + col ]
 *
 * Build requirements:
 *   -std=c++17   (minimum; -std=c++20 enables std::span / concepts)
 *   -O2 -march=native   (allows compiler SIMD auto-vectorisation of row ops)
 */

#pragma once

// ── Standard Library ────────────────────────────────────────────────────────
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

// ── C++20 Conditional Features ───────────────────────────────────────────────
#if __cplusplus >= 202002L
#  include <span>
#  include <concepts>
#  define OPTENG_CPP20 1
#else
#  define OPTENG_CPP20 0
#endif

// ── Compiler Hints ───────────────────────────────────────────────────────────
#if defined(__GNUC__) || defined(__clang__)
#  define OPTENG_LIKELY(x)   __builtin_expect(!!(x), 1)
#  define OPTENG_UNLIKELY(x) __builtin_expect(!!(x), 0)
#  define OPTENG_FORCE_INLINE __attribute__((always_inline)) inline
#else
#  define OPTENG_LIKELY(x)   (x)
#  define OPTENG_UNLIKELY(x) (x)
#  define OPTENG_FORCE_INLINE inline
#endif

// ────────────────────────────────────────────────────────────────────────────
namespace OptEngine {
// ────────────────────────────────────────────────────────────────────────────


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §1  COMPILE-TIME CONFIGURATION                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

namespace config {

    /// Magnitude of Big-M penalty. Must dominate every legitimate cost.
    /// Production note: computeAdaptiveBigM() scales this to the problem.
    inline constexpr double BIG_M_DEFAULT    = 1.0e7;

    /// Floating-point zero for pivot eligibility (|a_ij| < EPSILON → skip).
    inline constexpr double EPSILON          = 1.0e-10;

    /// Acceptable violation of b_i ≥ 0 after a pivot.
    inline constexpr double FEASIBILITY_TOL  = 1.0e-8;

    /// Reduced cost threshold for optimality: c̄_j ≥ −OPTIMALITY_TOL → done.
    inline constexpr double OPTIMALITY_TOL   = 1.0e-8;

    /// Degenerate ratio test guard: θ values within DEGENERACY_TOL are tied.
    inline constexpr double DEGENERACY_TOL   = 1.0e-12;

    /// Hard ceiling on simplex iterations (prevents infinite cycling).
    inline constexpr int    MAX_ITERATIONS   = 50'000;

    /// Positive-infinity sentinel (ratio test initialisation, INF-cost cells).
    inline constexpr double INF              = std::numeric_limits<double>::infinity();

    /// NaN sentinel for un-assigned transportation cost cells.
    inline constexpr double UNASSIGNED       = std::numeric_limits<double>::quiet_NaN();

    /// Degenerate allocation epsilon (perturbs zero-supply/demand allocations).
    inline constexpr double DEGENERATE_EPS   = 1.0e-14;

} // namespace config


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §2  ENUMERATIONS                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ---------------------------------------------------------------------------
/// Terminal state of any solver run.
/// Stored as uint8_t: packs into cache-line-friendly result structs.
// ---------------------------------------------------------------------------
enum class SolverStatus : uint8_t {
    NOT_STARTED       = 0,  ///< Solver not yet invoked
    OPTIMAL           = 1,  ///< Global optimum found
    INFEASIBLE        = 2,  ///< Feasible set is empty
    UNBOUNDED         = 3,  ///< Objective improves without bound
    MAX_ITER_REACHED  = 4,  ///< Hit config::MAX_ITERATIONS ceiling
    DEGENERATE        = 5,  ///< Cycling detected; Bland's rule engaged
    NUMERICAL_ERROR   = 6,  ///< Pivot collapse, NaN/Inf propagation
    PARTIAL           = 7,  ///< VAM BFS found; MODI phase not yet run
};

// ---------------------------------------------------------------------------
/// LP constraint type — drives the standard-form augmentation logic.
// ---------------------------------------------------------------------------
enum class ConstraintType : uint8_t {
    LESS_EQ    = 0,  ///< a^T x ≤ b  →  add slack   s  (s ≥ 0)
    GREATER_EQ = 1,  ///< a^T x ≥ b  →  subtract surplus s, add artificial a
    EQUAL      = 2,  ///< a^T x  = b  →  add artificial a only
};

/// Objective sense.  Internally always minimised; MAX flips sign of c.
enum class ObjectiveType : uint8_t {
    MINIMIZE = 0,
    MAXIMIZE = 1,
};

/// Pivot column selection strategy.
enum class PivotRule : uint8_t {
    MOST_NEGATIVE_RC = 0,   ///< Dantzig: argmin c̄_j (fastest convergence avg)
    BLANDS_RULE      = 1,   ///< Anti-cycling: smallest-index j with c̄_j < 0
    STEEPEST_EDGE    = 2,   ///< Max Δz per unit step (expensive; best iter count)
};

/// Column identity in the augmented tableau.
enum class VarType : uint8_t {
    ORIGINAL   = 0,  ///< Decision variable from original problem
    SLACK      = 1,  ///< Slack variable (≤ constraint)
    SURPLUS    = 2,  ///< Surplus variable (≥ constraint)
    ARTIFICIAL = 3,  ///< Artificial variable (≥ or = constraint)
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §3  EXCEPTION HIERARCHY                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/// Root exception for all OptEngine errors (inherits std::runtime_error).
class OptEngineException : public std::runtime_error {
public:
    explicit OptEngineException(std::string_view msg)
        : std::runtime_error(std::string(msg)) {}
};

/// Pivot denominator collapsed below EPSILON — numerical breakdown.
class NumericalInstabilityException final : public OptEngineException {
public:
    explicit NumericalInstabilityException(std::string_view detail)
        : OptEngineException(std::string("Numerical instability: ") +
                             std::string(detail)) {}
};

/// Artificial variables cannot be driven to zero — problem has no feasible point.
class InfeasibleProblemException final : public OptEngineException {
public:
    explicit InfeasibleProblemException(
        std::string_view detail = "Problem is infeasible — no feasible solution exists")
        : OptEngineException(detail) {}
};

/// Ratio test yields no finite leaving variable — objective unbounded below.
class UnboundedProblemException final : public OptEngineException {
public:
    explicit UnboundedProblemException(
        std::string_view detail = "Problem is unbounded — objective has no finite minimum")
        : OptEngineException(detail) {}
};

/// Matrix or vector sizes are mutually inconsistent.
class DimensionMismatchException final : public OptEngineException {
public:
    DimensionMismatchException(std::size_t expected, std::size_t got,
                               std::string_view context = "")
        : OptEngineException(
              std::string("Dimension mismatch") +
              (context.empty() ? "" : " in " + std::string(context)) +
              ": expected " + std::to_string(expected) +
              ", got "      + std::to_string(got)) {}
};

/// Transportation supply ≠ demand and automatic balancing was disabled.
class TransportationImbalanceException final : public OptEngineException {
public:
    TransportationImbalanceException(double supply, double demand)
        : OptEngineException(
              "Transportation imbalance: total supply (" +
              std::to_string(supply) + ") ≠ total demand (" +
              std::to_string(demand) + ")") {}
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §4  MATRIX — CACHE-OPTIMAL FLATTENED ROW-MAJOR STORAGE                ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @class Matrix
 * @brief  Contiguous row-major matrix backed by a single std::vector<double>.
 *
 * ┌─ Physical Memory (single heap block) ──────────────────────────────────┐
 * │                                                                          │
 * │  base+0        base+8       base+16      ...   base+(R·C-1)·8           │
 * │  ┌────────┬────────┬────────┬────────┬────────┬────────┐                │
 * │  │ [0][0] │ [0][1] │ [0][2] │ [1][0] │ [1][1] │  ...  │                │
 * │  └────────┴────────┴────────┴────────┴────────┴────────┘                │
 * │  ╠═══ Row 0 ════════════════╣╠═══ Row 1 ════════════════╣ ...           │
 * │                                                                          │
 * │  element(r, c)  =  data_[ r * cols_ + c ]                               │
 * │                                                                          │
 * │  Cache line (64 B) holds 8 doubles → row-sequential access is           │
 * │  stride-1 with hardware-prefetcher coverage after the first miss.       │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 * Design rationale:
 *   The simplex pivot kernel is dominated by row_axpy():
 *       row_i  ←  row_i  −  factor × row_j
 *   Both rows are contiguous → the CPU sees a linear scan of two memory
 *   regions → hardware prefetcher eliminates latency → enables SIMD
 *   auto-vectorisation (8 doubles/cycle on AVX-512).
 *   A std::vector<std::vector<double>> cannot achieve this because each
 *   inner vector is an independent heap allocation (see §Why-Not document).
 */
class Matrix {
public:

    // ── Construction ─────────────────────────────────────────────────────────

    Matrix() noexcept = default;

    /**
     * @brief  Zero-initialise a rows × cols matrix.
     * @throws std::invalid_argument  if rows == 0 || cols == 0.
     */
    explicit Matrix(std::size_t rows, std::size_t cols);

    /**
     * @brief  Construct from a pre-built flat vector (row-major order).
     * @throws DimensionMismatchException  if data.size() ≠ rows × cols.
     */
    Matrix(std::size_t rows, std::size_t cols, std::vector<double> data);

    Matrix(const Matrix&)                = default;
    Matrix& operator=(const Matrix&)     = default;
    Matrix(Matrix&&) noexcept            = default;
    Matrix& operator=(Matrix&&) noexcept = default;
    ~Matrix()                            = default;

    // ── Element Access ────────────────────────────────────────────────────────

    /**
     * @brief  Unchecked element access — the hot path.
     *         UB if row ≥ rows_ or col ≥ cols_.
     *         Inlined to a single address computation + load/store.
     */
    OPTENG_FORCE_INLINE double& operator()(std::size_t row,
                                           std::size_t col) noexcept {
        return data_[row * cols_ + col];
    }

    OPTENG_FORCE_INLINE double operator()(std::size_t row,
                                          std::size_t col) const noexcept {
        return data_[row * cols_ + col];
    }

    /**
     * @brief  Bounds-checked access.
     * @throws std::out_of_range on invalid indices.
     */
    [[nodiscard]] double& at(std::size_t row, std::size_t col);
    [[nodiscard]] double  at(std::size_t row, std::size_t col) const;

    // ── Row Pointer / Span Access ─────────────────────────────────────────────

    /**
     * @brief  Raw pointer to the first element of row r.
     *
     * Usage pattern (maximally cache-friendly):
     *   double* p = mat.row_ptr(r);
     *   for (std::size_t c = 0; c < mat.cols(); ++c) { process(p[c]); }
     *
     * This exposes a stride-1 linear scan of cols_ doubles — the compiler
     * can auto-vectorise the loop body with no aliasing concerns.
     */
    [[nodiscard]] OPTENG_FORCE_INLINE
    double* row_ptr(std::size_t row) noexcept {
        return data_.data() + row * cols_;
    }

    [[nodiscard]] OPTENG_FORCE_INLINE
    const double* row_ptr(std::size_t row) const noexcept {
        return data_.data() + row * cols_;
    }

#if OPTENG_CPP20
    /**
     * @brief  C++20 std::span view of a single row (zero-copy, range-safe).
     *         Enables structured binding and range-for without any copy.
     */
    [[nodiscard]] std::span<double>
    row_span(std::size_t row) noexcept {
        return { row_ptr(row), cols_ };
    }

    [[nodiscard]] std::span<const double>
    row_span(std::size_t row) const noexcept {
        return { row_ptr(row), cols_ };
    }
#endif

    // ── Dimension Queries ─────────────────────────────────────────────────────

    [[nodiscard]] std::size_t rows()  const noexcept { return rows_; }
    [[nodiscard]] std::size_t cols()  const noexcept { return cols_; }
    [[nodiscard]] std::size_t size()  const noexcept { return data_.size(); }
    [[nodiscard]] bool        empty() const noexcept { return data_.empty(); }

    // ── Raw Storage Access (BLAS / LAPACK / MKL interop) ─────────────────────

    [[nodiscard]] double*       data() noexcept       { return data_.data(); }
    [[nodiscard]] const double* data() const noexcept { return data_.data(); }

    // ── In-Place Row Operations ───────────────────────────────────────────────

    /**
     * @brief  DAXPY-equivalent row operation: row[target] -= factor * row[source]
     *
     * This is the fundamental kernel of Gaussian elimination and the simplex
     * pivot.  Operates on two contiguous memory segments of length cols_.
     * At -O2 with AVX2: processes 4 doubles/cycle; AVX-512: 8 doubles/cycle.
     *
     * @param target_row  Row to be updated (≠ source_row)
     * @param factor      Scalar multiplier
     * @param source_row  Row to subtract from target_row
     */
    void row_axpy(std::size_t target_row,
                  double      factor,
                  std::size_t source_row) noexcept;

    /**
     * @brief  Scale an entire row: row[r] *= scalar.
     *         Used during pivot normalisation (divide pivot row by pivot element).
     */
    void row_scale(std::size_t row, double scalar) noexcept;

    /**
     * @brief  Swap two rows via std::swap on contiguous ranges.
     *         O(cols_) — L1-cache-resident when cols_ ≤ ~512 doubles.
     */
    void swap_rows(std::size_t row_a, std::size_t row_b) noexcept;

    /**
     * @brief  Append one zero-initialised column to the right.
     *
     * ⚠ COST: O(rows × cols) data movement — row-major layout requires
     *   re-interleaving every existing row to insert a new slot.
     *   Prefer pre-allocating the full column count via the constructor.
     *   This method exists for iterative standard-form construction where
     *   column count is not known at construction time.
     */
    void append_column(double fill_value = 0.0);

    /**
     * @brief  Append one row (length must equal cols_).
     *         O(cols_) — trivially cheap for row-major: push_back cols_ doubles.
     * @throws DimensionMismatchException if row_data.size() ≠ cols_.
     */
    void append_row(std::vector<double> row_data);

    /**
     * @brief  Resize to new_rows × new_cols, preserving top-left overlap.
     *         New cells zero-initialised.
     */
    void resize(std::size_t new_rows, std::size_t new_cols);

    /**
     * @brief  Fill every element with value.
     */
    void fill(double value) noexcept;

    /**
     * @brief  Set entire matrix to 0.0 (semantics: memset-equivalent).
     */
    void zero() noexcept;

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /**
     * @brief  Pretty-print to stream (diagnostics only — not on the hot path).
     * @param precision  Decimal places (default 6)
     */
    void print(std::ostream& os, int precision = 6) const;

    /**
     * @brief  Assert internal invariants: rows_ * cols_ == data_.size().
     * @throws std::logic_error on corruption.
     */
    void validate() const;

private:
    std::size_t         rows_{ 0 };
    std::size_t         cols_{ 0 };
    std::vector<double> data_{};    ///< THE single contiguous 1-D storage block
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §5  PROBLEM DEFINITION STRUCTURES                                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @brief  One linear constraint:  coeffs · x  [≤ | ≥ | =]  rhs
 */
struct Constraint {
    std::vector<double> coeffs;   ///< Length = num_original_vars
    double              rhs;      ///< Right-hand side
    ConstraintType      type;     ///< LESS_EQ, GREATER_EQ, EQUAL
    std::string         label;    ///< Optional identifier for reporting

    Constraint(std::vector<double> c,
               double              b,
               ConstraintType      t,
               std::string         lbl = "")
        : coeffs(std::move(c)), rhs(b), type(t), label(std::move(lbl)) {}
};

/**
 * @brief  Complete LP specification in the user's natural form.
 *
 *   min / max   c^T x
 *   subject to  Constraint_i  for i = 0..m-1
 *               x ≥ 0  (non-negativity implicit)
 *
 * BigMSolver translates this to augmented standard form internally.
 */
struct LPProblem {
    std::vector<double>     objective;    ///< c — objective coefficients (length n)
    std::vector<Constraint> constraints;  ///< m constraints
    ObjectiveType           sense;        ///< MINIMIZE or MAXIMIZE
    std::string             name;         ///< Problem identifier (logging)
    double                  big_m;        ///< Override 0 → use config::BIG_M_DEFAULT

    explicit LPProblem(std::string   problem_name  = "LP",
                       ObjectiveType obj_sense      = ObjectiveType::MINIMIZE,
                       double        custom_big_m   = 0.0)
        : sense(obj_sense),
          name(std::move(problem_name)),
          big_m(custom_big_m > 0.0 ? custom_big_m : config::BIG_M_DEFAULT) {}

    /// Fluent interface: problem.add_constraint({1,2}, 10, LESS_EQ, "c1")
    LPProblem& add_constraint(std::vector<double> coeffs,
                               double              rhs,
                               ConstraintType      type,
                               std::string         label = "");

    /// Cross-validate: all constraint.coeffs.size() == objective.size().
    void validate() const;

    [[nodiscard]] std::size_t num_vars()        const noexcept { return objective.size(); }
    [[nodiscard]] std::size_t num_constraints() const noexcept { return constraints.size(); }
};

/**
 * @brief  Balanced transportation problem with m sources and n destinations.
 *
 * cost(i, j)  = unit shipping cost from source i to destination j.
 * supply[i]   = available units at source i.
 * demand[j]   = required units at destination j.
 *
 * Invariant (after balance()):  Σ supply = Σ demand.
 * The cost matrix is stored in the flattened Matrix class (row-major),
 * so iterating over all destinations of source i is stride-1.
 */
struct TransportationProblem {
    Matrix              cost;     ///< m×n unit cost matrix (row-major)
    std::vector<double> supply;   ///< m-length supply vector
    std::vector<double> demand;   ///< n-length demand vector
    std::string         name;

    TransportationProblem() = default;

    /**
     * @brief  Construct and optionally auto-balance the problem.
     * @param auto_balance  If true, adds dummy row or col when supply ≠ demand.
     * @throws TransportationImbalanceException if auto_balance is false and
     *         supply ≠ demand (within config::FEASIBILITY_TOL).
     */
    TransportationProblem(Matrix              cost_matrix,
                          std::vector<double> supply_vec,
                          std::vector<double> demand_vec,
                          std::string         problem_name = "TP",
                          bool                auto_balance = true);

    [[nodiscard]] std::size_t num_sources()      const noexcept { return supply.size(); }
    [[nodiscard]] std::size_t num_destinations() const noexcept { return demand.size(); }

    [[nodiscard]] double total_supply() const noexcept;
    [[nodiscard]] double total_demand() const noexcept;
    [[nodiscard]] bool   is_balanced()  const noexcept;

    /**
     * @brief  Append a dummy source (row) or dummy destination (column)
     *         with zero cost to make supply == demand.
     */
    void balance();

    void validate() const;
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §6  SIMPLEX TABLEAU — AUGMENTED SYSTEM STATE                           ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @brief  Per-column metadata for the augmented tableau.
 *         Enables full traceability from tableau column → original problem.
 */
struct ColumnDescriptor {
    VarType     type;           ///< ORIGINAL / SLACK / SURPLUS / ARTIFICIAL
    std::size_t source_index;   ///< Index in original problem vectors
    std::string label;          ///< "x1", "s2", "e3", "a4" etc.

    ColumnDescriptor(VarType t, std::size_t idx, std::string lbl)
        : type(t), source_index(idx), label(std::move(lbl)) {}
};

/**
 * @class SimplexTableau
 * @brief  Complete state of the augmented simplex system.
 *
 * ┌─ Tableau physical layout  (m+1 rows) × (n_aug+1 cols) ───────────────┐
 * │                                                                          │
 * │  Row 0     │ A_aug[0,*]  ·  ·  ·  A_aug[0,n_aug-1]  │  b[0]         │
 * │  Row 1     │ A_aug[1,*]  ·  ·  ·  A_aug[1,n_aug-1]  │  b[1]         │
 * │   ⋮        │      ⋮                     ⋮            │    ⋮          │
 * │  Row m-1   │ A_aug[m-1,*] · · · A_aug[m-1,n_aug-1]  │  b[m-1]       │
 * │  ──────────┼─────────────────────────────────────────┼───────────────│
 * │  Row m(OBJ)│   c̄[0]   · · ·    c̄[n_aug-1]         │  −z           │
 * │                                                                          │
 * │  All (m+1)×(n_aug+1) doubles in a SINGLE contiguous block via Matrix.  │
 * └──────────────────────────────────────────────────────────────────────────┘
 *
 * Key identity: reduced_cost(j)  ≡  tableau(num_constraints(), j)
 *               rhs(i)           ≡  tableau(i, num_augmented_vars())
 *
 * n_aug = num_original + num_slack + num_surplus + num_artificial
 */
struct SimplexTableau {

    // ── Core Contiguous Storage ───────────────────────────────────────────────
    Matrix tableau;             ///< (m+1) × (n_aug+1) augmented matrix

    // ── Basis Bookkeeping ─────────────────────────────────────────────────────
    std::vector<int>  basic_vars;
    ///< basic_vars[i] = column index of the variable in the basis for row i.
    ///< Length = m.  Invariant: column basic_vars[i] is the i-th identity
    ///< column in the current tableau.

    std::vector<int>  non_basic_vars;
    ///< Column indices of all non-basic (out-of-basis) variables.
    ///< Length = n_aug - m.  Value is 0 for non-basic variables.

    // ── Column Metadata ───────────────────────────────────────────────────────
    std::vector<ColumnDescriptor> column_info;
    ///< One entry per augmented column [0 .. n_aug-1].
    ///< Enables: "which tableau column corresponds to surplus variable #3?"

    std::vector<double> original_obj_coeffs;
    ///< c_j for original variables BEFORE Big-M inflation.
    ///< Used for objective value extraction (strip M·a_k terms from z).

    // ── Dimension Invariants ──────────────────────────────────────────────────
    std::size_t num_original_vars   { 0 };
    std::size_t num_slack_vars      { 0 };
    std::size_t num_surplus_vars    { 0 };
    std::size_t num_artificial_vars { 0 };

    /// Total augmented variable count = sum of all four above.
    [[nodiscard]] std::size_t num_augmented_vars() const noexcept {
        return num_original_vars + num_slack_vars +
               num_surplus_vars  + num_artificial_vars;
    }

    /// Number of constraint rows (excludes objective row).
    [[nodiscard]] std::size_t num_constraints() const noexcept {
        return (tableau.rows() > 0) ? tableau.rows() - 1 : 0;
    }

    /// Column index of the RHS (b / −z) column.
    [[nodiscard]] std::size_t rhs_col() const noexcept {
        return num_augmented_vars();
    }

    // ── Inline Hot-Path Accessors ─────────────────────────────────────────────

    /// Current RHS value for constraint row i (b_i after pivots).
    [[nodiscard]] OPTENG_FORCE_INLINE
    double rhs(std::size_t row) const noexcept {
        return tableau(row, rhs_col());
    }

    /// Reduced cost c̄_j: objective row value for column j.
    [[nodiscard]] OPTENG_FORCE_INLINE
    double reduced_cost(std::size_t col) const noexcept {
        return tableau(num_constraints(), col);
    }

    // ── Solution State ────────────────────────────────────────────────────────
    double       obj_value       { 0.0 };
    int          iteration_count { 0 };
    SolverStatus status          { SolverStatus::NOT_STARTED };
    bool         is_maximization { false };  ///< True → original was MAX, c was negated

    // ── Big-M Metadata ────────────────────────────────────────────────────────
    double           big_m_value           { config::BIG_M_DEFAULT };
    std::vector<int> artificial_col_indices; ///< Indices of all artificial columns
    ///< After solve: each must satisfy rhs(i) < config::FEASIBILITY_TOL
    ///< for the solution to be primal-feasible.

    // ── Lifecycle ─────────────────────────────────────────────────────────────
    SimplexTableau()                                     = default;
    SimplexTableau(const SimplexTableau&)                = default;
    SimplexTableau& operator=(const SimplexTableau&)     = default;
    SimplexTableau(SimplexTableau&&) noexcept            = default;
    SimplexTableau& operator=(SimplexTableau&&) noexcept = default;

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Full-tableau pretty-print with basis labels and reduced costs.
    void print(std::ostream& os, int precision = 6) const;

    /**
     * @brief  Validate structural invariants (debug builds):
     *   1. tableau.rows() == num_constraints() + 1
     *   2. tableau.cols() == num_augmented_vars() + 1
     *   3. basic_vars.size() == num_constraints()
     *   4. column_info.size() == num_augmented_vars()
     *   5. No NaN or Inf anywhere in tableau data.
     * @throws std::logic_error on any violation.
     */
    void validate() const;
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §7  SOLUTION RESULT STRUCTURES                                          ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @brief  Full LP solution package returned by BigMSolver::solve().
 */
struct LPSolution {
    SolverStatus        status;
    std::vector<double> variable_values;   ///< x*_j for original vars (length n)
    double              objective_value;   ///< Optimal z* (or INF if not optimal)
    int                 iteration_count;
    std::vector<double> dual_variables;    ///< Shadow prices y* (length m)
    std::vector<double> reduced_costs;     ///< c̄_j for original variables (length n)
    std::string         status_message;    ///< Human-readable outcome description

    explicit LPSolution(SolverStatus s = SolverStatus::NOT_STARTED)
        : status(s), objective_value(config::INF), iteration_count(0) {}

    [[nodiscard]] bool is_optimal()    const noexcept { return status == SolverStatus::OPTIMAL; }
    [[nodiscard]] bool is_infeasible() const noexcept { return status == SolverStatus::INFEASIBLE; }
    [[nodiscard]] bool is_unbounded()  const noexcept { return status == SolverStatus::UNBOUNDED; }
};

/// One allocation record in a transportation BFS or optimal plan.
struct AllocationCell {
    std::size_t source;       ///< Row index (source i)
    std::size_t destination;  ///< Column index (destination j)
    double      amount;       ///< x_ij — units shipped
    double      unit_cost;    ///< c_ij — cost per unit for this lane

    [[nodiscard]] double lane_cost() const noexcept { return amount * unit_cost; }
};

/**
 * @brief  Full transportation solution package returned by VogelSolver::solve().
 */
struct TransportationSolution {
    SolverStatus                status;
    std::vector<AllocationCell> allocations;       ///< Non-zero (basic) cells
    Matrix                      allocation_matrix; ///< Full m×n allocation grid
    double                      total_cost;
    int                         iteration_count;
    bool                        is_degenerate;     ///< #basics < m+n-1 at optimum
    std::string                 status_message;

    explicit TransportationSolution(SolverStatus s = SolverStatus::NOT_STARTED)
        : status(s), allocation_matrix(),
          total_cost(config::INF), iteration_count(0), is_degenerate(false) {}

    [[nodiscard]] bool is_optimal() const noexcept {
        return status == SolverStatus::OPTIMAL;
    }
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §8  BigMSolver — LINEAR PROGRAMMING VIA THE BIG-M METHOD               ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @class BigMSolver
 * @brief  Solves general LP problems (≤, ≥, =, mixed) using Big-M simplex.
 *
 * ┌─ Algorithm Pipeline ────────────────────────────────────────────────────┐
 * │                                                                          │
 * │  1. convertToStandardForm()                                              │
 * │     ≤  → +slack s_i                     (free basis column)             │
 * │     ≥  → -surplus e_i  +artificial a_i  (artificial enters basis)       │
 * │     =  →              +artificial a_i   (artificial enters basis)        │
 * │     MAX → negate c                       (transform to min)              │
 * │     b_i < 0 → flip row sign + flip type (enforce b ≥ 0 invariant)       │
 * │                                                                          │
 * │  2. initializeTableau()                                                  │
 * │     Embed A_aug into Matrix, set up basis as {slacks ∪ artificials},    │
 * │     apply applyBigMPenalties() to restore canonical obj row form.        │
 * │                                                                          │
 * │  3. Main loop (≤ MAX_ITERATIONS):                                        │
 * │     a. findPivotColumn() → entering variable j (or nullopt → OPTIMAL)   │
 * │     b. findPivotRow()    → leaving variable i  (or throw UNBOUNDED)     │
 * │     c. performPivot()    → row_scale + n×row_axpy on Matrix             │
 * │                                                                          │
 * │  4. checkArtificialFeasibility() → throw INFEASIBLE if a_k > TOL        │
 * │  5. extractPrimalSolution() · extractDualSolution() · extractReducedCosts│
 * └──────────────────────────────────────────────────────────────────────────┘
 */
class BigMSolver {
public:

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief  Construct solver for a given LP problem.
     * @param problem   Fully specified LPProblem (validated immediately).
     * @param rule      Pivot column selection strategy.
     * @throws OptEngineException if problem.validate() fails.
     */
    explicit BigMSolver(LPProblem problem,
                        PivotRule rule = PivotRule::MOST_NEGATIVE_RC);

    BigMSolver(const BigMSolver&)            = delete;  ///< No copy — owns mutable state
    BigMSolver& operator=(const BigMSolver&) = delete;
    BigMSolver(BigMSolver&&)                 = default;
    BigMSolver& operator=(BigMSolver&&)      = default;
    ~BigMSolver()                            = default;

    // ── Primary Interface ─────────────────────────────────────────────────────

    /**
     * @brief  Execute the complete Big-M simplex solve.
     *
     * Calls: convertToStandardForm → initializeTableau → pivot loop →
     *        checkArtificialFeasibility → extractSolution.
     *
     * @return LPSolution containing status, x*, z*, duals, reduced costs.
     * @throws InfeasibleProblemException        (status = INFEASIBLE)
     * @throws UnboundedProblemException         (status = UNBOUNDED)
     * @throws NumericalInstabilityException     (status = NUMERICAL_ERROR)
     */
    [[nodiscard]] LPSolution solve();

    /**
     * @brief  Re-optimise with a new objective vector, reusing the current basis.
     *
     * When only c changes:
     *   1. Recompute reduced costs:  c̄ = c - c_B · B^{-1} · A
     *   2. If all c̄ ≥ 0: current BFS is still optimal → return immediately.
     *   3. Else run the dual simplex (or re-pivot from current basis).
     *   This avoids re-running Phase I (standard form + Big-M initialisation).
     *
     * @param new_obj   Must have size == problem_.num_vars().
     * @throws DimensionMismatchException if new_obj.size() ≠ num_vars().
     */
    [[nodiscard]] LPSolution resolve_with_new_objective(std::vector<double> new_obj);

    // ── Standard Form Construction ────────────────────────────────────────────

    /**
     * @brief  Translate user-space problem into augmented standard form.
     *
     * Post-condition (guarantee for initializeTableau):
     *   std_form_A_  : m × n_aug coefficient matrix (as row vectors)
     *   std_form_b_  : m-length RHS vector with all b_i ≥ 0
     *   std_form_c_  : n_aug-length cost vector (M applied to artificials)
     *   tableau_.column_info : fully populated
     *
     * @pre  problem_.validate() passed.
     * @throws DimensionMismatchException  if any constraint length ≠ n.
     */
    void convertToStandardForm();

    /**
     * @brief  Build the SimplexTableau from the standard-form data.
     *
     * Steps:
     *   1. Allocate Matrix((m+1), (n_aug+1)) — single contiguous block.
     *   2. Fill constraint rows from std_form_A_ and std_form_b_.
     *   3. Fill objective row from std_form_c_ (objective coefficients).
     *   4. Set basic_vars to initial basis (slacks / artificials).
     *   5. Call applyBigMPenalties() to zero out artificial basis columns
     *      in the objective row (canonical form requirement).
     *
     * @pre  convertToStandardForm() called successfully.
     */
    void initializeTableau();

    // ── Pivot Operations ──────────────────────────────────────────────────────

    /**
     * @brief  Identify the entering variable (pivot column).
     *
     * MOST_NEGATIVE_RC:  argmin_{j} c̄_j   where c̄_j < -OPTIMALITY_TOL
     * BLANDS_RULE:       smallest index j   where c̄_j < -OPTIMALITY_TOL
     * STEEPEST_EDGE:     argmax_{j} |Δz_j / ‖d_j‖|  (most improvement / step)
     *
     * @param tab  Current tableau (read-only).
     * @return     Column index of entering variable, or std::nullopt if optimal.
     */
    [[nodiscard]] std::optional<std::size_t>
    findPivotColumn(const SimplexTableau& tab) const noexcept;

    /**
     * @brief  Identify the leaving variable (pivot row) via minimum ratio test.
     *
     * Ratio test:  θ* = min { b_i / a_{i,q}  :  a_{i,q} > EPSILON }
     *                    i
     * Degeneracy:  Multiple rows achieve θ* = 0.
     *   → Resolve by Bland's tie-break: smallest basic_vars[i] among tied rows.
     *   → Alternatively: Charnes' lexicographic perturbation (see private helpers).
     *
     * @param tab        Current tableau (read-only).
     * @param pivot_col  Entering column index q.
     * @return           Row index of leaving variable, or std::nullopt if
     *                   all a_{i,q} ≤ EPSILON (objective is UNBOUNDED).
     * @throws UnboundedProblemException  directly when no positive a_{i,q} exists.
     */
    [[nodiscard]] std::optional<std::size_t>
    findPivotRow(const SimplexTableau& tab,
                 std::size_t           pivot_col) const;

    /**
     * @brief  Execute one complete simplex pivot on the tableau.
     *
     * Sequence of row operations on the Matrix object:
     *   Step 1. Normalise:  row_scale(pivot_row, 1/a_{p,q})
     *                       → pivot element becomes 1.0
     *   Step 2. Eliminate:  for each i ≠ pivot_row:
     *                           row_axpy(i, -a_{i,q}, pivot_row)
     *                       → entire pivot column becomes identity vector e_p
     *   Step 3. Update basis:
     *                       basic_vars[pivot_row]     = pivot_col (entering)
     *                       non_basic_vars[leaving_i] = old basic_vars[pivot_row]
     *
     * Each row operation is a stride-1 scan of n_aug+1 doubles — the
     * hardware prefetcher achieves steady-state throughput from Step 2 onward.
     *
     * @param tab        Tableau to mutate (basic_vars and Matrix updated).
     * @param pivot_row  Row of leaving variable (p).
     * @param pivot_col  Column of entering variable (q).
     * @throws NumericalInstabilityException if |a_{p,q}| < EPSILON.
     */
    void performPivot(SimplexTableau& tab,
                      std::size_t     pivot_row,
                      std::size_t     pivot_col);

    // ── Termination Checks ────────────────────────────────────────────────────

    /**
     * @brief  Optimality check: all reduced costs ≥ -OPTIMALITY_TOL.
     * O(n_aug) scan of objective row — cache-friendly (row = contiguous).
     */
    [[nodiscard]] bool isOptimal(const SimplexTableau& tab) const noexcept;

    /**
     * @brief  Feasibility check after Big-M optimisation:
     *         any artificial a_k in basis with value > FEASIBILITY_TOL
     *         → original problem is INFEASIBLE.
     * @throws InfeasibleProblemException on detection.
     */
    [[nodiscard]] bool
    checkArtificialFeasibility(const SimplexTableau& tab) const;

    // ── Solution Extraction ───────────────────────────────────────────────────

    /**
     * @brief  Extract primal solution vector x* from the optimal tableau.
     *
     * For j in [0, num_original_vars):
     *   If j ∈ basic_vars: x*_j = rhs(row where basic_vars[row] == j)
     *   Else (non-basic):  x*_j = 0.0
     *
     * Complexity: O(m) scan of basic_vars.
     *
     * @return  x* vector of length num_original_vars.
     */
    [[nodiscard]] std::vector<double>
    extractPrimalSolution(const SimplexTableau& tab) const;

    /**
     * @brief  Extract dual solution (shadow prices) y* = c_B B^{-1}.
     *
     * From the optimal tableau, for constraint i with slack variable s_i
     * (column k_i), the shadow price is:
     *   y*_i = − c̄_{k_i}   (negative of slack's reduced cost in obj row)
     *
     * For ≥ / = constraints (no slack), extract from the artificial column's
     * reduced cost and negate the sign convention.
     *
     * @return  y* of length num_constraints().
     */
    [[nodiscard]] std::vector<double>
    extractDualSolution(const SimplexTableau& tab) const;

    /**
     * @brief  Extract reduced costs c̄_j for all original variables.
     * @return  c̄ of length num_original_vars (objective row values).
     */
    [[nodiscard]] std::vector<double>
    extractReducedCosts(const SimplexTableau& tab) const;

    // ── Big-M Penalty Infrastructure ─────────────────────────────────────────

    /**
     * @brief  Apply Big-M penalties to the objective row for initial basis.
     *
     * The simplex tableau must be in canonical form w.r.t. the current basis:
     * the objective row coefficient for each basic variable must be 0.
     * For artificials in the initial basis with cost M, we subtract M × row_k
     * from the objective row for each artificial k.
     *
     * Formula applied for each artificial a_k in basis at row r_k:
     *   obj_row ← obj_row − M × constraint_row_{r_k}
     *
     * This is identical in structure to a pivot elimination step on the obj row.
     *
     * @param tab  Tableau (objective row mutated in place).
     */
    void applyBigMPenalties(SimplexTableau& tab) const;

    /**
     * @brief  Compute an adaptive Big-M coefficient from the objective.
     *
     * Fixed-large-M (e.g. 1e7) causes numerical catastrophe when legitimate
     * cost magnitudes are ≫ 1 (e.g. costs in millions):
     *   - Reduced costs become dominated by M, masking true optimality structure
     *   - Pivot selection distorted → extra iterations or incorrect optimality
     *
     * Adaptive heuristic:
     *   M = max(1, max_j |c_j|) × 10^4
     *   clamped to [config::BIG_M_DEFAULT, 1e12]
     *
     * @param obj_coeffs  Original objective coefficient vector.
     * @return  Scaled Big-M value safe for this problem's magnitude.
     */
    [[nodiscard]] static double
    computeAdaptiveBigM(const std::vector<double>& obj_coeffs) noexcept;

    // ── State Inspection ──────────────────────────────────────────────────────

    [[nodiscard]] const SimplexTableau& getTableau() const noexcept { return tableau_; }
    [[nodiscard]] SolverStatus          getStatus()  const noexcept { return tableau_.status; }

    /// Print full tableau with reduced costs to stream.
    void printTableau(std::ostream& os, int precision = 6) const;

private:
    // ── Owned Problem State ───────────────────────────────────────────────────
    LPProblem      problem_;        ///< Immutable original specification
    SimplexTableau tableau_;        ///< Mutable working tableau
    PivotRule      pivot_rule_;

    // ── Standard Form Intermediate Buffers ───────────────────────────────────
    std::vector<std::vector<double>> std_form_A_; ///< Augmented rows (m × n_aug)
    std::vector<double>              std_form_b_; ///< Non-negative RHS
    std::vector<double>              std_form_c_; ///< Augmented objective with Big-M

    // ── Private Helpers ───────────────────────────────────────────────────────

    /**
     * @brief  Enforce b_i ≥ 0:  if b_i < 0, multiply row by −1 and flip type.
     *         (LESS_EQ ↔ GREATER_EQ,  EQUAL stays EQUAL with negated RHS)
     *         Simplex ratio test requires non-negative RHS at initialisation.
     */
    void enforceNonNegativeRHS();

    /// Query: does column col_idx belong to an artificial variable?
    [[nodiscard]] bool isArtificial(std::size_t col_idx) const noexcept;

    /// Bland's rule: return smallest-index j with c̄_j < -OPTIMALITY_TOL.
    [[nodiscard]] std::optional<std::size_t>
    blandsPivotColumn(const SimplexTableau& tab) const noexcept;

    /**
     * @brief  Charnes' ε-perturbation: add iε^k to b_k for each basic row k.
     *         Converts degenerate BFS to lexicographically non-degenerate,
     *         guaranteeing finite termination without Bland's rule overhead.
     */
    void perturbDegenerateRHS(SimplexTableau& tab) const noexcept;
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §9  VogelSolver — TRANSPORTATION PROBLEM (VAM + MODI)                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝

/**
 * @class VogelSolver
 * @brief  Solves balanced transportation problems to optimality.
 *
 * ┌─ Two-Phase Pipeline ────────────────────────────────────────────────────┐
 * │                                                                          │
 * │  PHASE 1 — Vogel's Approximation Method (VAM):                          │
 * │  ─────────────────────────────────────────────                          │
 * │  Goal: find a high-quality initial basic feasible solution (BFS)        │
 * │        with exactly m+n−1 basic cells.                                  │
 * │                                                                          │
 * │  Iterate until all supply and demand exhausted:                          │
 * │    1. computeRowPenalties() : p_i = 2nd_min_cost(row i) - min_cost(i)  │
 * │    2. computeColPenalties() : p_j = 2nd_min_cost(col j) - min_cost(j)  │
 * │    3. findMaxPenaltyDimension() : select row or col with max p          │
 * │    4. findMinCostIn{Row,Col}()  : find cheapest uneliminated cell       │
 * │    5. allocate() : x_ij = min(supply_i, demand_j); update remaining    │
 * │    6. Eliminate row i or col j (or both if degenerate)                  │
 * │                                                                          │
 * │  PHASE 2 — MODI (Modified Distribution / UV) Method:                   │
 * │  ──────────────────────────────────────────────────                     │
 * │  Goal: verify optimality and improve until all Δ_ij ≥ 0.               │
 * │                                                                          │
 * │  Iterate until optimal:                                                  │
 * │    1. computeUVValues()        : solve u_i + v_j = c_ij for basics     │
 * │    2. computeOpportunityCosts(): Δ_ij = c_ij - u_i - v_j for non-basics│
 * │    3. isMODIOptimal()          : all Δ_ij ≥ -OPTIMALITY_TOL → DONE     │
 * │    4. findEnteringCell()       : argmin Δ_ij (most negative)            │
 * │    5. findImprovementLoop()    : trace closed loop through basic cells  │
 * │    6. improveAllocation()      : θ* = min of odd-position cells        │
 * │                                  even cells += θ*,  odd cells -= θ*    │
 * └──────────────────────────────────────────────────────────────────────────┘
 */
class VogelSolver {
public:

    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief  Construct solver, auto-balancing if supply ≠ demand.
     * @throws TransportationImbalanceException if problem cannot be balanced.
     */
    explicit VogelSolver(TransportationProblem problem);

    VogelSolver(const VogelSolver&)            = delete;
    VogelSolver& operator=(const VogelSolver&) = delete;
    VogelSolver(VogelSolver&&)                 = default;
    VogelSolver& operator=(VogelSolver&&)      = default;
    ~VogelSolver()                             = default;

    // ── Primary Interface ─────────────────────────────────────────────────────

    /**
     * @brief  Run VAM to find BFS, then MODI to optimality.
     * @return TransportationSolution containing allocation matrix and total cost.
     */
    [[nodiscard]] TransportationSolution solve();

    // ── Phase 1: Vogel's Approximation Method ────────────────────────────────

    /**
     * @brief  Compute row opportunity penalties for the current reduced problem.
     *
     * For active (non-eliminated) row i:
     *   penalty_i = (2nd cheapest cost among active cols) − (cheapest cost)
     *
     * Interpretation: if we are denied the cheapest cell in row i, we are
     * penalised by penalty_i per unit.  VAM prioritises the highest-penalty
     * row/column to limit the maximum regret at each step.
     *
     * @param remaining_supply  Current residual supply (length m).
     * @param remaining_demand  Current residual demand (length n).
     * @param eliminated_rows   Flags for rows with supply = 0.
     * @param eliminated_cols   Flags for cols with demand = 0.
     * @return  Per-row penalty vector (length m).  Eliminated rows → INF.
     */
    [[nodiscard]] std::vector<double>
    computeRowPenalties(const std::vector<double>& remaining_supply,
                        const std::vector<double>& remaining_demand,
                        const std::vector<bool>&   eliminated_rows,
                        const std::vector<bool>&   eliminated_cols) const;

    /**
     * @brief  Compute column opportunity penalties (symmetric to row version).
     *
     * penalty_j = min2_j - min1_j  over active rows.
     *
     * @return  Per-column penalty vector (length n).  Eliminated cols → INF.
     */
    [[nodiscard]] std::vector<double>
    computeColPenalties(const std::vector<double>& remaining_supply,
                        const std::vector<double>& remaining_demand,
                        const std::vector<bool>&   eliminated_rows,
                        const std::vector<bool>&   eliminated_cols) const;

    /**
     * @brief  Identify the row or column with the maximum penalty.
     *
     * Tie-breaking: compare max row penalty vs max column penalty.
     * If equal: select the dimension with the absolute minimum cost in
     * the dominant row/col (Reinfeld & Vogel original heuristic).
     *
     * @param row_penalties  Output of computeRowPenalties().
     * @param col_penalties  Output of computeColPenalties().
     * @return  {is_row, index}:
     *            is_row = true  → returned index is a row index
     *            is_row = false → returned index is a column index
     */
    [[nodiscard]] std::pair<bool, std::size_t>
    findMaxPenaltyDimension(const std::vector<double>& row_penalties,
                            const std::vector<double>& col_penalties) const noexcept;

    /**
     * @brief  Find the column index of the minimum-cost active cell in row r.
     * Ties: smallest column index.
     * @param row_idx         Source row.
     * @param eliminated_cols Active column mask.
     * @return  Column index j* = argmin_{j active} cost(row_idx, j).
     */
    [[nodiscard]] std::size_t
    findMinCostInRow(std::size_t              row_idx,
                     const std::vector<bool>& eliminated_cols) const noexcept;

    /**
     * @brief  Find the row index of the minimum-cost active cell in column c.
     * @param col_idx         Destination column.
     * @param eliminated_rows Active row mask.
     * @return  Row index i* = argmin_{i active} cost(i, col_idx).
     */
    [[nodiscard]] std::size_t
    findMinCostInCol(std::size_t              col_idx,
                     const std::vector<bool>& eliminated_rows) const noexcept;

    /**
     * @brief  Perform one VAM allocation at cell (row_idx, col_idx).
     *
     * Allocated amount = min(remaining_supply[row_idx], remaining_demand[col_idx])
     * Updates:
     *   allocation(row_idx, col_idx)  +=  amount
     *   remaining_supply[row_idx]     −=  amount
     *   remaining_demand[col_idx]     −=  amount
     *
     * Elimination logic:
     *   supply exhausted → eliminated_rows[row_idx] = true
     *   demand satisfied → eliminated_cols[col_idx] = true
     *   BOTH (degenerate): eliminate row; set allocation(row_idx, col_idx)
     *                      to DEGENERATE_EPS (maintains m+n-1 basic count).
     *
     * @return  AllocationCell record for this step.
     */
    AllocationCell
    allocate(std::size_t          row_idx,
             std::size_t          col_idx,
             Matrix&              allocation,
             std::vector<double>& remaining_supply,
             std::vector<double>& remaining_demand,
             std::vector<bool>&   eliminated_rows,
             std::vector<bool>&   eliminated_cols);

    // ── Phase 2: MODI (UV Method) ─────────────────────────────────────────────

    /**
     * @brief  Solve the UV system u_i + v_j = c_ij for all basic cells.
     *
     * Formulation: basic cells form a spanning tree on the bipartite graph
     * (source-nodes × destination-nodes).  The UV equations are solvable by
     * BFS traversal of this tree, starting with u_0 = 0.
     *
     * Degeneracy handling: if the spanning tree is disconnected (fewer than
     * m+n-1 basic cells), the system is underdetermined.  repairDegeneracy()
     * must be called first.
     *
     * @param allocation   Current allocation matrix (identifies basic cells).
     * @param basic_cells  Ordered list of basic allocations.
     * @param u            Output: u_i values (length m), initialised to 0.
     * @param v            Output: v_j values (length n), initialised to UNASSIGNED.
     */
    void computeUVValues(const Matrix&                      allocation,
                         const std::vector<AllocationCell>& basic_cells,
                         std::vector<double>&               u,
                         std::vector<double>&               v) const;

    /**
     * @brief  Compute opportunity costs Δ_ij = c_ij − u_i − v_j.
     *
     * Δ_ij < 0  →  shipping via (i,j) reduces total cost → (i,j) should enter basis.
     * Δ_ij ≥ 0  →  current allocation is optimal for this cell.
     * Basic cells: set to +INF (they are in the basis, not candidates).
     *
     * @param allocation  Allocation matrix (basic cell detection).
     * @param u           u_i values from computeUVValues().
     * @param v           v_j values from computeUVValues().
     * @return  m×n Matrix of Δ_ij values.
     */
    [[nodiscard]] Matrix
    computeOpportunityCosts(const Matrix&              allocation,
                            const std::vector<double>& u,
                            const std::vector<double>& v) const;

    /**
     * @brief  MODI optimality check: all Δ_ij ≥ -OPTIMALITY_TOL for non-basics.
     * @return  true  → current BFS is globally optimal.
     */
    [[nodiscard]] bool
    isMODIOptimal(const Matrix& delta_matrix) const noexcept;

    /**
     * @brief  Find the most negative Δ_ij (entering non-basic cell).
     * Ties: smallest (row, col) lexicographic index.
     * @return  {row, col} of argmin Δ_ij over non-basic cells.
     */
    [[nodiscard]] std::pair<std::size_t, std::size_t>
    findEnteringCell(const Matrix& delta_matrix) const noexcept;

    /**
     * @brief  Trace the unique closed loop through the entering cell.
     *
     * A transportation loop is a sequence of cells:
     *   (r_0,c_0) → (r_0,c_1) → (r_1,c_1) → (r_1,c_0) → ... → (r_0,c_0)
     * Properties:
     *   - (r_0,c_0) is the entering non-basic cell.
     *   - All other cells are basic.
     *   - Alternates strictly between horizontal and vertical moves.
     *   - Has even length ≥ 4.
     *   - Unique for non-degenerate BFS (uniqueness of spanning tree path).
     *
     * Implementation: backtracking DFS on bipartite basic-cell graph.
     *
     * @param allocation    Current allocation matrix.
     * @param entering_row  Row of entering cell (r_0).
     * @param entering_col  Column of entering cell (c_0).
     * @return  Ordered loop as vector of {row, col} pairs.
     * @throws  OptEngineException  if no loop found (degenerate failure).
     */
    [[nodiscard]] std::vector<std::pair<std::size_t, std::size_t>>
    findImprovementLoop(const Matrix& allocation,
                        std::size_t   entering_row,
                        std::size_t   entering_col) const;

    /**
     * @brief  Execute one MODI improvement step along a closed loop.
     *
     * Convention:
     *   Even-indexed loop cells (0, 2, 4, …): allocation += θ*
     *   Odd-indexed loop cells  (1, 3, 5, …): allocation −= θ*
     *   θ* = min { allocation(r_k, c_k) : k odd }
     *
     * The odd cell achieving θ* leaves the basis.
     * Degenerate θ* = 0: the cell with the smallest (row, col) index
     * among the minimising odd cells is chosen as the leaving cell
     * (added to basis as degenerate basic with value 0).
     *
     * @param allocation  Allocation Matrix (mutated in place).
     * @param loop        Loop from findImprovementLoop().
     */
    void improveAllocation(
        Matrix&                                                   allocation,
        const std::vector<std::pair<std::size_t, std::size_t>>&  loop);

    // ── Degeneracy Handling ───────────────────────────────────────────────────

    /**
     * @brief  Detect degenerate BFS: #basic_cells < m + n − 1.
     * @param basic_cells  Current basic allocations.
     * @return  true if degenerate.
     */
    [[nodiscard]] bool
    isDegenerateBFS(const std::vector<AllocationCell>& basic_cells) const noexcept;

    /**
     * @brief  Repair degeneracy: add DEGENERATE_EPS to minimum-cost
     *         unoccupied cells until exactly m+n-1 basic cells exist.
     *
     * Cells added at near-zero cost minimise objective distortion.
     * The epsilon values are treated as zero in the final solution
     * extraction (filtered by allocation > DEGENERATE_EPS threshold).
     *
     * @param allocation   Allocation matrix (mutated).
     * @param basic_cells  Basic cell list (mutated by appending new entries).
     */
    void repairDegeneracy(Matrix&                      allocation,
                          std::vector<AllocationCell>& basic_cells) const;

    // ── Solution Assembly ─────────────────────────────────────────────────────

    /**
     * @brief  Compute total cost: Σ_{i,j} cost(i,j) × allocation(i,j).
     * O(m×n) scan — fully sequential on the row-major Matrix.
     */
    [[nodiscard]] double
    computeTotalCost(const Matrix& allocation) const noexcept;

    /**
     * @brief  Extract non-zero allocations from the full m×n matrix.
     * Filters entries with amount > DEGENERATE_EPS to exclude epsilon fixes.
     */
    [[nodiscard]] std::vector<AllocationCell>
    extractAllocations(const Matrix& allocation) const;

    // ── Diagnostics ───────────────────────────────────────────────────────────

    void printAllocation(const Matrix& allocation, std::ostream& os) const;
    void printDeltaMatrix(const Matrix& delta,     std::ostream& os) const;

    [[nodiscard]] const TransportationProblem& getProblem() const noexcept {
        return problem_;
    }

private:
    TransportationProblem problem_;    ///< Working copy (potentially auto-balanced)
    bool                  balanced_;   ///< Was the problem already balanced on input?

    /// Allocate and zero-initialise the m×n allocation matrix.
    [[nodiscard]] Matrix initAllocationMatrix() const;

    /// Extract basic cells: entries with allocation > DEGENERATE_EPS.
    [[nodiscard]] std::vector<AllocationCell>
    extractBasicCells(const Matrix& allocation) const;

    /**
     * @brief  BFS over the bipartite spanning tree of basic cells.
     * Used by computeUVValues() to propagate u/v values through the tree.
     * Nodes: {0..m-1} are source nodes, {m..m+n-1} are destination nodes.
     */
    void traverseSpanningTree(
        const std::vector<AllocationCell>& basic_cells,
        std::vector<double>&               u,
        std::vector<double>&               v) const;
};


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  §10  UTILITY NAMESPACE                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════╝

namespace utils {

    /**
     * @brief  Dot product of equal-length vectors: Σ a_i b_i.
     * @throws DimensionMismatchException if sizes differ.
     */
    [[nodiscard]] double dot_product(const std::vector<double>& a,
                                     const std::vector<double>& b);

    /**
     * @brief  L∞ norm: max |v_i|.  Returns 0.0 for empty vector.
     */
    [[nodiscard]] double linf_norm(const std::vector<double>& v) noexcept;

    /**
     * @brief  L1 norm: Σ |v_i|.
     */
    [[nodiscard]] double l1_norm(const std::vector<double>& v) noexcept;

    /// Test if |x| ≤ tol.
    [[nodiscard]] OPTENG_FORCE_INLINE
    bool is_zero(double x, double tol = config::EPSILON) noexcept {
        return std::abs(x) <= tol;
    }

    /// Test a ≤ b + tol (for robust ≤ comparison in ratio tests).
    [[nodiscard]] OPTENG_FORCE_INLINE
    bool leq(double a, double b, double tol = config::EPSILON) noexcept {
        return a <= b + tol;
    }

    /// Test |a − b| ≤ tol (floating-point equality).
    [[nodiscard]] OPTENG_FORCE_INLINE
    bool approx_equal(double a, double b,
                      double tol = config::EPSILON) noexcept {
        return std::abs(a - b) <= tol;
    }

    /// Human-readable label for SolverStatus.
    [[nodiscard]] std::string_view status_to_string(SolverStatus s) noexcept;

    /// Human-readable label for ConstraintType.
    [[nodiscard]] std::string_view constraint_type_to_string(ConstraintType t) noexcept;

    /**
     * @brief  Validate that every element of v is ≥ 0 (supply/demand/RHS check).
     * @throws std::invalid_argument on negative entry (reports index).
     */
    void validate_non_negative(const std::vector<double>& v,
                               std::string_view            name);

    /**
     * @brief  Print a labeled vector to stream at given precision.
     */
    void print_vector(const std::vector<double>& v,
                      std::string_view             label,
                      std::ostream&               os,
                      int                          precision = 6);

    /**
     * @brief  Check for NaN or Inf in a vector (numerical guard).
     * @return  true if any element is NaN or Inf.
     */
    [[nodiscard]] bool has_numerical_issue(const std::vector<double>& v) noexcept;

    /**
     * @brief  Index of the minimum element in a vector (optionally masked).
     * @param v     Source vector.
     * @param mask  If provided (same size), only unmasked (false) elements considered.
     * @return  Index of minimum, or std::nullopt if all masked.
     */
    [[nodiscard]] std::optional<std::size_t>
    argmin(const std::vector<double>& v,
           const std::vector<bool>*   mask = nullptr) noexcept;

    /**
     * @brief  Index of the maximum element in a vector (optionally masked).
     */
    [[nodiscard]] std::optional<std::size_t>
    argmax(const std::vector<double>& v,
           const std::vector<bool>*   mask = nullptr) noexcept;

} // namespace utils

// ────────────────────────────────────────────────────────────────────────────
} // namespace OptEngine
// ────────────────────────────────────────────────────────────────────────────
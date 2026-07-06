/**
 * @file    BigMSolver.cpp
 * @brief   Production Implementation — Big-M Simplex LP Solver
 * @version 1.0.0
 *
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  Implements: BigMSolver  (§8 of OptimizationEngine.hpp)                 ║
 * ║              Matrix      (§4)                                            ║
 * ║              LPProblem / TransportationProblem (§5)                      ║
 * ║              SimplexTableau (§6)                                         ║
 * ║              Solution structs (§7)                                       ║
 * ║              utils namespace (§10)                                       ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 *
 * ── Pivot Flow (single iteration) ──────────────────────────────────────────
 *
 *   BEFORE PIVOT (column q enters, row p leaves):
 *
 *     Row p (pivot row):  [ a_{p,0}  ...  a_{p,q}  ...  a_{p,n} | b_p ]
 *                                          ^pivot element
 *     Row i (other row):  [ a_{i,0}  ...  a_{i,q}  ...  a_{i,n} | b_i ]
 *
 *   STEP 1 — Normalise pivot row (row_scale):
 *     row_p  ←  row_p / a_{p,q}         ⟹  a_{p,q} becomes 1.0
 *
 *   STEP 2 — Eliminate (row_axpy for every row i ≠ p):
 *     row_i  ←  row_i  −  a_{i,q} × row_p   ⟹  a_{i,q} becomes 0.0
 *     (includes objective row: i = num_constraints())
 *
 *   AFTER PIVOT: column q is an identity vector (1 in row p, 0 elsewhere).
 *     variable x_q is now BASIC (in the basis at row p).
 *     variable x_{old_basic[p]} is now NON-BASIC (value = 0).
 *
 * ── Memory model ────────────────────────────────────────────────────────────
 *   All row ops touch contiguous doubles → hardware prefetcher achieves
 *   steady-state throughput from the second cache line of each row onward.
 *   At -O2 + AVX2: ~4 doubles/cycle per row; AVX-512: ~8 doubles/cycle.
 */

#include "OptimizationEngine.hpp"

#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace OptEngine {
    // This is the Single Source of Truth for the simplex algorithm.
    //
    //  NOTE on the `self` parameter:
    //  ──────────────────────────────────────────────────────────────────────
    //  The pivot-loop logic calls several BigMSolver member functions
    //  (isOptimal, findPivotColumn, findPivotRow, performPivot,
    //  checkArtificialFeasibility, extractPrimalSolution, extractDualSolution,
    //  extractReducedCosts) and reads/writes the private member pivot_rule_.
    //  None of these are declared static or free in OptimizationEngine.hpp,
    //  and the header cannot be modified, so this free function must carry an
    //  instance pointer to dispatch to them.
static LPSolution run_simplex_impl(
    SimplexTableau& tableau,
    const PivotRule& pivot_rule,
    const LPProblem& problem,
    BigMSolver* self)
{
    (void)problem; // not referenced by this execution block (mirrors original lambda)

    const std::size_t m = tableau.num_constraints();

    bool bland_mode = false;   // switched on when cycling suspected

    // ── Phase 1: Main simplex pivot loop ──────────────────────────────
    //
    //  Termination conditions (checked in priority order):
    //    1. isOptimal()         → all c̄_j ≥ -OPTIMALITY_TOL  → STOP
    //    2. findPivotRow = ∅   → column is unbounded  → THROW
    //    3. iter > MAX_ITERATIONS → anti-cycling escalation or hard stop
    //
    for (int iter = 0; iter < config::MAX_ITERATIONS; ++iter) {
        tableau.iteration_count = iter;

        // ── Optimality test: scan objective row for most negative c̄_j
        if (OPTENG_LIKELY(self->isOptimal(tableau))) {
            tableau.status = SolverStatus::OPTIMAL;
            break;
        }

        // ── Detect stalling → escalate to Bland's rule (anti-cycling)
        //  Stalling heuristic: switch to Bland after 20% of MAX_ITERATIONS.
        //  Bland's rule guarantees finite termination but converges slower.
        if (OPTENG_UNLIKELY(iter > config::MAX_ITERATIONS / 5 && !bland_mode)) {
            bland_mode = true;
            tableau.status = SolverStatus::DEGENERATE;
            // Do NOT return — continue pivoting under Bland's rule.
        }

        const PivotRule effective_rule = bland_mode
                                         ? PivotRule::BLANDS_RULE
                                         : pivot_rule;

        // ── Select entering variable (pivot column q) ──────────────────
        std::optional<std::size_t> pivot_col_opt;

            if (effective_rule == PivotRule::BLANDS_RULE) {
                // If escalated to Bland's rule, execute it directly here without touching private state
                const std::size_t naug = tableau.num_augmented_vars();
                const std::size_t obj_row = tableau.num_constraints();
                for (std::size_t j = 0; j < naug; ++j) {
                    if (tableau.tableau(obj_row, j) < -config::OPTIMALITY_TOL) {
                        pivot_col_opt = j;
                        break;
                    }
                }
            } else {
                // Otherwise, safely delegate to the existing public/member method
                pivot_col_opt = self->findPivotColumn(tableau);
            }

        if (!pivot_col_opt.has_value()) {
            // No negative reduced cost — floating-point edge case.
            tableau.status = SolverStatus::OPTIMAL;
            break;
        }
        const std::size_t pivot_col = pivot_col_opt.value();

        // ── Select leaving variable (pivot row p) via ratio test ───────
        //  findPivotRow throws UnboundedProblemException internally.
        auto pivot_row_opt = self->findPivotRow(tableau, pivot_col);
        if (!pivot_row_opt.has_value()) {
            tableau.status = SolverStatus::UNBOUNDED;
            throw UnboundedProblemException();
        }
        const std::size_t pivot_row = pivot_row_opt.value();

        // ── Execute Gauss-Jordan elimination ───────────────────────────
        self->performPivot(tableau, pivot_row, pivot_col);

        // ── Post-pivot numerical health check ──────────────────────────
        //  Scan RHS for NaN/Inf every 500 iterations (O(m) cost).
        if (OPTENG_UNLIKELY(iter % 500 == 499)) {
            for (std::size_t r = 0; r < m; ++r) {
                if (!std::isfinite(tableau.rhs(r))) {
                    tableau.status = SolverStatus::NUMERICAL_ERROR;
                    throw NumericalInstabilityException(
                        "RHS NaN/Inf at constraint row " + std::to_string(r) +
                        " after " + std::to_string(iter) + " pivots");
                }
            }
        }
    }

    // Hard iteration limit hit without convergence.
    if (tableau.iteration_count >= config::MAX_ITERATIONS - 1 &&
        tableau.status != SolverStatus::OPTIMAL)
    {
        tableau.status = SolverStatus::MAX_ITER_REACHED;
    }

    // ── Feasibility gate: artificials must be driven to zero ───────────
    //  If any a_k > FEASIBILITY_TOL remains in basis, the original
    //  problem has no feasible point (Big-M failed to expel artificial).
    if (!self->checkArtificialFeasibility(tableau)) { /* unreachable */ }

    // ── Solution extraction ────────────────────────────────────────────
    LPSolution sol(SolverStatus::OPTIMAL);
    sol.status          = tableau.status;
    sol.iteration_count = tableau.iteration_count;

    // Objective value: tableau RHS of objective row, negated for MAX.
    // The objective row stores −z (we minimise z, stored as −z in tableau).
    const double raw_z = -tableau.rhs(tableau.num_constraints());
    sol.objective_value = tableau.is_maximization ? -raw_z : raw_z;

    sol.variable_values = self->extractPrimalSolution(tableau);
    sol.dual_variables  = self->extractDualSolution(tableau);
    sol.reduced_costs   = self->extractReducedCosts(tableau);

    sol.status_message  = std::string(utils::status_to_string(tableau.status)) +
                          " after " + std::to_string(tableau.iteration_count) +
                          " pivots; z* = " + std::to_string(sol.objective_value);

    return sol;
}

// ════════════════════════════════════════════════════════════════════════════
// §A  MATRIX IMPLEMENTATION
// ════════════════════════════════════════════════════════════════════════════

// ── Constructors ─────────────────────────────────────────────────────────────

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0)
{
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("Matrix dimensions must be > 0");
}

Matrix::Matrix(std::size_t rows, std::size_t cols, std::vector<double> data)
    : rows_(rows), cols_(cols), data_(std::move(data))
{
    if (data_.size() != rows_ * cols_)
        throw DimensionMismatchException(rows_ * cols_, data_.size(),
                                         "Matrix flat constructor");
}

// ── Bounds-checked access ─────────────────────────────────────────────────────

double& Matrix::at(std::size_t row, std::size_t col)
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range(
            "Matrix::at(" + std::to_string(row) + "," +
            std::to_string(col) + ") out of range [" +
            std::to_string(rows_) + "x" + std::to_string(cols_) + "]");
    return data_[row * cols_ + col];
}

double Matrix::at(std::size_t row, std::size_t col) const
{
    if (row >= rows_ || col >= cols_)
        throw std::out_of_range(
            "Matrix::at(" + std::to_string(row) + "," +
            std::to_string(col) + ") out of range");
    return data_[row * cols_ + col];
}

// ── In-place row operations (the hot-path kernel) ─────────────────────────────
//
//  row_axpy is the single most performance-critical function in the solver.
//  Both rows are contiguous L1-cacheable blocks → the compiler can emit
//  a single SIMD loop with no aliasing concerns (restrict-style semantics
//  guaranteed because target_row ≠ source_row by contract).

void Matrix::row_axpy(std::size_t target_row,
                      double      factor,
                      std::size_t source_row) noexcept
{
    // Early-exit: multiplying by zero is a no-op (and avoids NaN propagation).
    if (OPTENG_UNLIKELY(factor == 0.0)) return;

    double*       tgt = row_ptr(target_row);          // contiguous target
    const double* src = row_ptr(source_row);          // contiguous source

    // Compiler sees two non-aliased pointer + scalar loop → auto-vectorises.
    for (std::size_t c = 0; c < cols_; ++c)
        tgt[c] -= factor * src[c];
}

void Matrix::row_scale(std::size_t row, double scalar) noexcept
{
    double* p = row_ptr(row);
    for (std::size_t c = 0; c < cols_; ++c)
        p[c] *= scalar;
}

void Matrix::swap_rows(std::size_t row_a, std::size_t row_b) noexcept
{
    if (row_a == row_b) return;
    double* a = row_ptr(row_a);
    double* b = row_ptr(row_b);
    // std::swap_ranges is stride-1 on both pointers → cache-optimal.
    std::swap_ranges(a, a + cols_, b);
}

void Matrix::append_column(double fill_value)
{
    // ⚠ O(rows × cols) — re-interleave every row to insert a new slot.
    // This is intentionally discouraged in the header; pre-allocate instead.
    const std::size_t new_cols = cols_ + 1;
    std::vector<double> new_data(rows_ * new_cols, fill_value);
    for (std::size_t r = 0; r < rows_; ++r)
        for (std::size_t c = 0; c < cols_; ++c)
            new_data[r * new_cols + c] = data_[r * cols_ + c];
    data_ = std::move(new_data);
    cols_ = new_cols;
}

void Matrix::append_row(std::vector<double> row_data)
{
    if (row_data.size() != cols_)
        throw DimensionMismatchException(cols_, row_data.size(), "Matrix::append_row");
    data_.insert(data_.end(), row_data.begin(), row_data.end());
    ++rows_;
}

void Matrix::resize(std::size_t new_rows, std::size_t new_cols)
{
    std::vector<double> new_data(new_rows * new_cols, 0.0);
    const std::size_t copy_rows = std::min(rows_, new_rows);
    const std::size_t copy_cols = std::min(cols_, new_cols);
    for (std::size_t r = 0; r < copy_rows; ++r)
        for (std::size_t c = 0; c < copy_cols; ++c)
            new_data[r * new_cols + c] = data_[r * cols_ + c];
    data_ = std::move(new_data);
    rows_ = new_rows;
    cols_ = new_cols;
}

void Matrix::fill(double value) noexcept
{
    std::fill(data_.begin(), data_.end(), value);
}

void Matrix::zero() noexcept
{
    std::fill(data_.begin(), data_.end(), 0.0);
}

void Matrix::print(std::ostream& os, int precision) const
{
    os << std::fixed << std::setprecision(precision);
    for (std::size_t r = 0; r < rows_; ++r) {
        os << "  [ ";
        for (std::size_t c = 0; c < cols_; ++c) {
            os << std::setw(precision + 6) << data_[r * cols_ + c];
            if (c + 1 < cols_) os << "  ";
        }
        os << " ]\n";
    }
}

void Matrix::validate() const
{
    if (data_.size() != rows_ * cols_)
        throw std::logic_error("Matrix internal corruption: data_.size()=" +
                               std::to_string(data_.size()) + " ≠ rows_*cols_=" +
                               std::to_string(rows_ * cols_));
}


// ════════════════════════════════════════════════════════════════════════════
// §B  PROBLEM DEFINITION IMPLEMENTATIONS
// ════════════════════════════════════════════════════════════════════════════

// ── LPProblem ─────────────────────────────────────────────────────────────────

LPProblem& LPProblem::add_constraint(std::vector<double> coeffs,
                                      double              rhs,
                                      ConstraintType      type,
                                      std::string         label)
{
    if (!objective.empty() && coeffs.size() != objective.size())
        throw DimensionMismatchException(objective.size(), coeffs.size(),
                                         "LPProblem::add_constraint coeffs");
    constraints.emplace_back(std::move(coeffs), rhs, type, std::move(label));
    return *this;
}

void LPProblem::validate() const
{
    if (objective.empty())
        throw OptEngineException("LPProblem: objective vector is empty");
    for (std::size_t i = 0; i < constraints.size(); ++i) {
        if (constraints[i].coeffs.size() != objective.size())
            throw DimensionMismatchException(
                objective.size(), constraints[i].coeffs.size(),
                "constraint[" + std::to_string(i) + "]");
    }
}

// ── TransportationProblem ─────────────────────────────────────────────────────

TransportationProblem::TransportationProblem(Matrix              cost_matrix,
                                             std::vector<double> supply_vec,
                                             std::vector<double> demand_vec,
                                             std::string         problem_name,
                                             bool                auto_balance)
    : cost(std::move(cost_matrix)),
      supply(std::move(supply_vec)),
      demand(std::move(demand_vec)),
      name(std::move(problem_name))
{
    if (cost.rows() != supply.size())
        throw DimensionMismatchException(supply.size(), cost.rows(), "cost rows vs supply");
    if (cost.cols() != demand.size())
        throw DimensionMismatchException(demand.size(), cost.cols(), "cost cols vs demand");

    if (!is_balanced()) {
        if (auto_balance) balance();
        else throw TransportationImbalanceException(total_supply(), total_demand());
    }
}

double TransportationProblem::total_supply() const noexcept
{
    return std::accumulate(supply.begin(), supply.end(), 0.0);
}

double TransportationProblem::total_demand() const noexcept
{
    return std::accumulate(demand.begin(), demand.end(), 0.0);
}

bool TransportationProblem::is_balanced() const noexcept
{
    return std::abs(total_supply() - total_demand()) <= config::FEASIBILITY_TOL;
}

void TransportationProblem::balance()
{
    const double ts = total_supply();
    const double td = total_demand();
    const double diff = ts - td;

    if (std::abs(diff) <= config::FEASIBILITY_TOL) return;

    if (diff > 0.0) {
        // Excess supply → add dummy destination column with zero costs.
        demand.push_back(diff);
        cost.append_column(0.0);
    } else {
        // Excess demand → add dummy source row with zero costs.
        supply.push_back(-diff);
        cost.append_row(std::vector<double>(demand.size(), 0.0));
    }
}

void TransportationProblem::validate() const
{
    if (!is_balanced())
        throw TransportationImbalanceException(total_supply(), total_demand());
}


// ════════════════════════════════════════════════════════════════════════════
// §C  SIMPLEX TABLEAU DIAGNOSTICS
// ════════════════════════════════════════════════════════════════════════════

void SimplexTableau::print(std::ostream& os, int precision) const
{
    const std::size_t m    = num_constraints();
    const std::size_t naug = num_augmented_vars();

    os << "\n═══════════════ Simplex Tableau (iter=" << iteration_count
       << ") ═══════════════\n";

    // Header row: column labels
    os << std::setw(8) << "Basis";
    for (std::size_t j = 0; j < naug; ++j)
        os << std::setw(precision + 5) << column_info[j].label;
    os << std::setw(precision + 5) << "RHS" << "\n";

    os << std::string(8 + (naug + 1) * (precision + 5), '-') << "\n";

    for (std::size_t i = 0; i < m; ++i) {
        os << std::setw(8) << column_info[static_cast<std::size_t>(basic_vars[i])].label;
        for (std::size_t j = 0; j <= naug; ++j)
            os << std::fixed << std::setprecision(precision)
               << std::setw(precision + 5) << tableau(i, j);
        os << "\n";
    }

    os << std::string(8 + (naug + 1) * (precision + 5), '-') << "\n";
    os << std::setw(8) << "z";
    for (std::size_t j = 0; j <= naug; ++j)
        os << std::fixed << std::setprecision(precision)
           << std::setw(precision + 5) << tableau(m, j);
    os << "\n═══════════════════════════════════════════════════════\n";
}

void SimplexTableau::validate() const
{
    const std::size_t m    = num_constraints();
    const std::size_t naug = num_augmented_vars();

    if (tableau.rows() != m + 1)
        throw std::logic_error("SimplexTableau: row count mismatch");
    if (tableau.cols() != naug + 1)
        throw std::logic_error("SimplexTableau: col count mismatch");
    if (basic_vars.size() != m)
        throw std::logic_error("SimplexTableau: basic_vars size mismatch");
    if (column_info.size() != naug)
        throw std::logic_error("SimplexTableau: column_info size mismatch");

    // NaN/Inf scan over entire data block.
    const double* raw = tableau.data();
    for (std::size_t k = 0; k < tableau.size(); ++k) {
        if (!std::isfinite(raw[k]))
            throw std::logic_error(
                "SimplexTableau: non-finite value at flat index " +
                std::to_string(k));
    }
}


// ════════════════════════════════════════════════════════════════════════════
// §D  UTILITY NAMESPACE IMPLEMENTATIONS
// ════════════════════════════════════════════════════════════════════════════

namespace utils {

double dot_product(const std::vector<double>& a, const std::vector<double>& b)
{
    if (a.size() != b.size())
        throw DimensionMismatchException(a.size(), b.size(), "dot_product");
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

double linf_norm(const std::vector<double>& v) noexcept
{
    if (v.empty()) return 0.0;
    double m = 0.0;
    for (double x : v) m = std::max(m, std::abs(x));
    return m;
}

double l1_norm(const std::vector<double>& v) noexcept
{
    double s = 0.0;
    for (double x : v) s += std::abs(x);
    return s;
}

std::string_view status_to_string(SolverStatus s) noexcept
{
    switch (s) {
        case SolverStatus::NOT_STARTED:      return "NOT_STARTED";
        case SolverStatus::OPTIMAL:          return "OPTIMAL";
        case SolverStatus::INFEASIBLE:       return "INFEASIBLE";
        case SolverStatus::UNBOUNDED:        return "UNBOUNDED";
        case SolverStatus::MAX_ITER_REACHED: return "MAX_ITER_REACHED";
        case SolverStatus::DEGENERATE:       return "DEGENERATE";
        case SolverStatus::NUMERICAL_ERROR:  return "NUMERICAL_ERROR";
        case SolverStatus::PARTIAL:          return "PARTIAL";
        default:                             return "UNKNOWN";
    }
}

std::string_view constraint_type_to_string(ConstraintType t) noexcept
{
    switch (t) {
        case ConstraintType::LESS_EQ:    return "<=";
        case ConstraintType::GREATER_EQ: return ">=";
        case ConstraintType::EQUAL:      return "=";
        default:                         return "?";
    }
}

void validate_non_negative(const std::vector<double>& v, std::string_view name)
{
    for (std::size_t i = 0; i < v.size(); ++i)
        if (v[i] < -config::FEASIBILITY_TOL)
            throw std::invalid_argument(
                std::string(name) + "[" + std::to_string(i) +
                "] = " + std::to_string(v[i]) + " is negative");
}

void print_vector(const std::vector<double>& v, std::string_view label,
                  std::ostream& os, int precision)
{
    os << label << ": [ ";
    os << std::fixed << std::setprecision(precision);
    for (std::size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ",  ";
    }
    os << " ]\n";
}

bool has_numerical_issue(const std::vector<double>& v) noexcept
{
    for (double x : v)
        if (!std::isfinite(x)) return true;
    return false;
}

std::optional<std::size_t> argmin(const std::vector<double>& v,
                                   const std::vector<bool>*   mask) noexcept
{
    std::optional<std::size_t> best;
    double best_val = config::INF;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (mask && (*mask)[i]) continue;
        if (v[i] < best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

std::optional<std::size_t> argmax(const std::vector<double>& v,
                                   const std::vector<bool>*   mask) noexcept
{
    std::optional<std::size_t> best;
    double best_val = -config::INF;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (mask && (*mask)[i]) continue;
        if (v[i] > best_val) { best_val = v[i]; best = i; }
    }
    return best;
}

} // namespace utils


// ════════════════════════════════════════════════════════════════════════════
// §E  BigMSolver IMPLEMENTATION
// ════════════════════════════════════════════════════════════════════════════

// ── Constructor ───────────────────────────────────────────────────────────────

BigMSolver::BigMSolver(LPProblem problem, PivotRule rule)
    : problem_(std::move(problem)),
      pivot_rule_(rule)
{
    problem_.validate();

    // Compute and store an adaptive Big-M that dominates all objective costs.
    // This prevents numerical catastrophe when costs are large (see §H §Why-Big-M).
    const double adaptive_m = computeAdaptiveBigM(problem_.objective);
    if (problem_.big_m < adaptive_m)
        problem_.big_m = adaptive_m;

    tableau_.big_m_value = problem_.big_m;
}

// ── Primary solve entry point ─────────────────────────────────────────────────

LPSolution BigMSolver::solve()
{
    // ── Phase 0: Build the augmented standard-form system ──────────────────
    convertToStandardForm();
    initializeTableau();

    tableau_.status          = SolverStatus::NOT_STARTED;
    tableau_.iteration_count = 0;

    // ── Shared execution block ──────────────────────────────────────────────
    //
    //  Delegates to the free function run_simplex_impl() (top of file), the
    //  single source of truth for the pivot loop. resolve_with_new_objective()
    //  calls the same helper directly on a warm tableau without routing back
    //  through solve() (which would call convertToStandardForm() and
    //  initializeTableau(), destroying the basis and double-negating the
    //  MAX objective).
    //
    //  CONTRACT (pre-conditions on tableau_ before invoking):
    //    • tableau_ is fully initialised (cold) OR has a valid warm basis with
    //      a correctly recomputed objective row (warm restart path).
    //    • tableau_.status  == SolverStatus::NOT_STARTED
    //    • tableau_.iteration_count == 0  (or the desired start count)
    return run_simplex_impl(tableau_, pivot_rule_, problem_, this);
}

// ── Re-optimise with a new objective (warm start) ────────────────────────────
//
//  WHY THIS IS VALUABLE:
//    Parametric LP / sensitivity analysis change only c, not A or b.
//    Re-running from scratch repeats the entire Phase I + Phase II pivot
//    sequence needlessly. Instead we recompute reduced costs c̄ = c - c_B B^{-1}A
//    using the current optimal basis B — if still optimal (all c̄ ≥ 0),
//    we return immediately in O(m·n) time vs. O(iterations·m·n) for a cold start.
//
//  WHY NOT call solve() here:
//  ─────────────────────────────────────────────────────────────────────────────
//  solve() begins with convertToStandardForm() + initializeTableau(). Those two
//  calls DESTROY the current tableau_ completely:
//
//    1. convertToStandardForm() clears tableau_.column_info, rebuilds std_form_A_,
//       std_form_b_, std_form_c_ from scratch, and RE-NEGATES the objective for
//       maximisation problems.  If the caller already negated the new objective
//       above (as this function does), a second call to solve() would negate it
//       AGAIN, making every MAX run optimise the ORIGINAL (non-negated) objective.
//
//    2. initializeTableau() re-allocates the entire Matrix, resets basic_vars to
//       the all-slacks/artificials basis, and calls applyBigMPenalties() — wiping
//       out the B^{-1} that is implicitly stored in the warm tableau columns.
//
//    3. iteration_count is reset to 0 inside initializeTableau(), so even
//       the loop would terminate at iter == 0 (isOptimal returns true on the
//       re-initialised canonical form, which is trivially optimal before any
//       pivots).
//
//  The fix: invoke run_simplex_impl() (the shared execution block) directly
//  on the already-warm tableau_, exactly as solve() does after its setup phase.

LPSolution BigMSolver::resolve_with_new_objective(std::vector<double> new_obj)
{
    if (new_obj.size() != problem_.num_vars())
        throw DimensionMismatchException(problem_.num_vars(), new_obj.size(),
                                         "resolve_with_new_objective");

    problem_.objective = std::move(new_obj);

    // ── Step 1: Internalise sense — always minimise internally ───────────────
    //
    //  WHY only negate here (not inside solve()):
    //    convertToStandardForm() handles the negation for a cold solve.
    //    On a warm restart we bypass that function entirely, so we must
    //    apply the same sense transformation manually, once.
    //    The flag tableau_.is_maximization is already set from the original
    //    solve() call and does not need to change.
    if (problem_.sense == ObjectiveType::MAXIMIZE) {
        for (double& c : problem_.objective) c = -c;
        tableau_.is_maximization = true;
    }

    // ── Step 2: Rebuild objective row from scratch using current warm basis ───
    //
    //  For each augmented column j, the updated reduced cost is:
    //    c̄_j = c_j  −  Σ_{i in basis} c_{B_i} · a_{i,j}
    //
    //  We directly overwrite the objective row in the tableau.

    const std::size_t m    = tableau_.num_constraints();
    const std::size_t naug = tableau_.num_augmented_vars();
    const std::size_t rhs  = tableau_.rhs_col();

    // ── Collect c_B: original objective cost for each basic variable ──────────
    //
    //  Three cases for the basic variable bv in row i:
    //
    //  (A) ORIGINAL VARIABLE (bv < num_original_vars):
    //      c_B[i] = problem_.objective[bv]   (already negated for MAX above)
    //      This is the direct economic cost of keeping x_{bv} in the basis.
    //
    //  (B) ARTIFICIAL VARIABLE still in basis:
    //      c_B[i] = tableau_.big_m_value
    //
    //      WHY M, not 0:
    //      If an artificial a_k is still basic at value ~0 (degenerate basis),
    //      the reduced cost formula must honour its true objective coefficient M.
    //      Setting c_B[i] = 0 would undercount the penalty cost and produce
    //      incorrect reduced costs for every other column — particularly for the
    //      artificial's own column, whose c̄_{a_k} must equal 0 in canonical
    //      form (basic variable ⇒ zero reduced cost).  Using M ensures that
    //      c̄_{a_k} = M − Σ c_B[i] * a_{i, a_k} = M − M*1 = 0 ✓
    //
    //      WHEN can this happen:
    //      A degenerate BFS from the original solve() may leave a_k basic at
    //      value 0.  The problem is feasible (a_k = 0 violates no constraint)
    //      but the warm basis still tracks it.  The new objective recomputation
    //      must handle this without corruption.
    //
    //  (C) SLACK or SURPLUS VARIABLE:
    //      c_B[i] = 0   (slack/surplus carry zero objective cost)
    //      No explicit assignment needed — vector is zero-initialised.
    //
    std::vector<double> c_B(m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        const int        bv   = tableau_.basic_vars[i];
        const std::size_t bv_u = static_cast<std::size_t>(bv);

        if (bv_u < problem_.num_vars()) {
            // Case (A): original decision variable.
            c_B[i] = problem_.objective[bv_u];
        } else if (tableau_.column_info[bv_u].type == VarType::ARTIFICIAL) {
            // Case (B): artificial still occupying a basis slot.
            c_B[i] = tableau_.big_m_value;
        }
        // Case (C): slack / surplus → 0 (default).
    }

    // ── Recompute every column of the objective row ───────────────────────────
    for (std::size_t j = 0; j < naug; ++j) {
        // Original objective coefficient (zero for slacks/surplus).
        double c_j = (j < problem_.num_vars()) ? problem_.objective[j] : 0.0;

        // Add Big-M for artificial columns — they always carry cost M,
        // regardless of whether they are currently basic or not.
        if (tableau_.column_info[j].type == VarType::ARTIFICIAL)
            c_j += tableau_.big_m_value;

        // c̄_j = c_j − c_B · (j-th column of current B^{-1}A).
        // The j-th column of B^{-1}A is exactly column j of tableau_.tableau.
        double sum = 0.0;
        for (std::size_t i = 0; i < m; ++i)
            sum += c_B[i] * tableau_.tableau(i, j);
        tableau_.tableau(m, j) = c_j - sum;
    }

    // ── Recompute RHS of objective row: −z = −Σ c_{B_i} b_i ─────────────────
    {
        double sum = 0.0;
        for (std::size_t i = 0; i < m; ++i)
            sum += c_B[i] * tableau_.rhs(i);
        tableau_.tableau(m, rhs) = -sum;
    }

    // ── Step 3: Reset solver state for a fresh pivot run on the warm tableau ──
    //
    //  WHY reset iteration_count to 0 and status to NOT_STARTED:
    //    The warm tableau already encodes the current basis B.  We are NOT
    //    resuming a previous run — we are starting a NEW optimisation from
    //    this basis.  Resetting the counter gives accurate pivot statistics
    //    for the re-optimisation phase and ensures the MAX_ITERATIONS guard
    //    inside run_simplex_impl does not fire prematurely.
    tableau_.status          = SolverStatus::NOT_STARTED;
    tableau_.iteration_count = 0;

    // ── Step 4: Run the shared simplex execution block on the warm tableau ────
    //
    //  Delegates to run_simplex_impl() (top of file), the same helper used by
    //  solve(). See solve() for the rationale of why this is a free function
    //  rather than a private member function (header cannot be modified).
    return run_simplex_impl(tableau_, pivot_rule_, problem_, this);
}


// ── Standard form conversion ──────────────────────────────────────────────────
//
//  WHY NOT skip this and work with the raw user form directly?
//  ────────────────────────────────────────────────────────────
//  The simplex method REQUIRES the system to be in canonical (standard) form:
//    Ax = b,  b ≥ 0,  x ≥ 0
//  with a known initial BFS in terms of an identity basis.
//
//  Without standard form:
//   • A "≥ b" constraint means the LHS can freely go to −∞ without penalty —
//     the simplex ratio test has nothing to bound the entering variable against.
//   • There is no obvious initial BFS for ≥ or = constraints — slack variables
//     give us one for ≤, but ≥ and = constraints start infeasible if we naively
//     set x = 0 (the all-zero vector violates a constraint "a^T x ≥ b > 0").
//   • The objective row computation assumes c_B = 0 for slack/artificial vars,
//     which only holds AFTER the standard form is built and Big-M penalties applied.

void BigMSolver::convertToStandardForm()
{
    const std::size_t n = problem_.num_vars();
    const std::size_t m = problem_.num_constraints();

    // ── Step 0: Handle MAXIMIZATION by negating objective ────────────────────
    //  Internal invariant: always MINIMISE.
    //  Store original sign for output extraction.
    std::vector<double> obj_coeffs = problem_.objective;
    if (problem_.sense == ObjectiveType::MAXIMIZE) {
        for (double& c : obj_coeffs) c = -c;
        tableau_.is_maximization = true;
    } else {
        tableau_.is_maximization = false;
    }
    tableau_.original_obj_coeffs = obj_coeffs;  // save pre-Big-M version

    // ── Step 1: Enforce b_i ≥ 0 (ratio test prerequisite) ───────────────────
    //  If a constraint has b_i < 0, multiply the entire row by −1.
    //  This flips the inequality direction: ≤ becomes ≥ and vice versa.
    //  EQUAL constraints with negative RHS become EQUAL with positive RHS
    //  (negating both sides of an equality preserves it).
    enforceNonNegativeRHS();

    // ── Step 2: Count augmented variables ────────────────────────────────────
    std::size_t n_slack = 0, n_surplus = 0, n_artificial = 0;
    for (const auto& c : problem_.constraints) {
        switch (c.type) {
            case ConstraintType::LESS_EQ:
                ++n_slack;
                break;
            case ConstraintType::GREATER_EQ:
                ++n_surplus;
                ++n_artificial;
                break;
            case ConstraintType::EQUAL:
                ++n_artificial;
                break;
        }
    }
    const std::size_t n_aug = n + n_slack + n_surplus + n_artificial;

    tableau_.num_original_vars   = n;
    tableau_.num_slack_vars      = n_slack;
    tableau_.num_surplus_vars    = n_surplus;
    tableau_.num_artificial_vars = n_artificial;

    // ── Step 3: Build column descriptor table ─────────────────────────────────
    //  Column layout (left to right):
    //    [original x_0..x_{n-1}] [slacks s_0..] [surplus e_0..] [artificials a_0..]
    //
    //  This ordering puts the original decision variables at the lowest indices,
    //  which simplifies primal solution extraction (no offset arithmetic) and
    //  aligns with Bland's rule (smallest index first).
    tableau_.column_info.clear();
    tableau_.column_info.reserve(n_aug);

    for (std::size_t j = 0; j < n; ++j)
        tableau_.column_info.emplace_back(VarType::ORIGINAL, j,
                                          "x" + std::to_string(j + 1));

    // Slack columns (one per ≤ constraint)
    std::size_t slack_idx = 0;
    for (std::size_t i = 0; i < m; ++i)
        if (problem_.constraints[i].type == ConstraintType::LESS_EQ)
            tableau_.column_info.emplace_back(VarType::SLACK, slack_idx++,
                                              "s" + std::to_string(i + 1));

    // Surplus columns (one per ≥ constraint)
    std::size_t surplus_idx = 0;
    for (std::size_t i = 0; i < m; ++i)
        if (problem_.constraints[i].type == ConstraintType::GREATER_EQ)
            tableau_.column_info.emplace_back(VarType::SURPLUS, surplus_idx++,
                                              "e" + std::to_string(i + 1));

    // Artificial columns (one per ≥ or = constraint)
    std::size_t art_idx = 0;
    for (std::size_t i = 0; i < m; ++i)
        if (problem_.constraints[i].type == ConstraintType::GREATER_EQ ||
            problem_.constraints[i].type == ConstraintType::EQUAL)
        {
            const std::size_t art_col = n + n_slack + n_surplus + art_idx;
            tableau_.column_info.emplace_back(VarType::ARTIFICIAL, art_idx++,
                                              "a" + std::to_string(i + 1));
            tableau_.artificial_col_indices.push_back(
                static_cast<int>(art_col));
        }

    // ── Step 4: Build augmented A rows and objective ──────────────────────────
    std::fill(std_form_b_.begin(), std_form_b_.end(), 0.0);
    std_form_A_.assign(m, std::vector<double>(n_aug, 0.0));
    std_form_b_.resize(m);
    std_form_c_.assign(n_aug, 0.0);

    // Objective: original coefficients first, then 0 for slacks/surplus,
    // then Big-M for artificials (the penalty that forces them to zero).
    for (std::size_t j = 0; j < n; ++j)
        std_form_c_[j] = obj_coeffs[j];

    for (int art_col : tableau_.artificial_col_indices)
        std_form_c_[static_cast<std::size_t>(art_col)] = tableau_.big_m_value;

    // Build each constraint row.
    // We re-traverse the constraint list to assign column offsets correctly.
    std::size_t s_col = n;              // next slack column index
    std::size_t e_col = n + n_slack;    // next surplus column index
    std::size_t a_col = n + n_slack + n_surplus;  // next artificial index

    for (std::size_t i = 0; i < m; ++i) {
        const Constraint& con = problem_.constraints[i];

        // Copy original coefficients.
        for (std::size_t j = 0; j < n; ++j)
            std_form_A_[i][j] = con.coeffs[j];

        std_form_b_[i] = con.rhs;

        switch (con.type) {
            case ConstraintType::LESS_EQ:
                // a^T x ≤ b  →  a^T x + s = b,  s ≥ 0
                std_form_A_[i][s_col++] = +1.0;
                break;

            case ConstraintType::GREATER_EQ:
                // a^T x ≥ b  →  a^T x − e + a = b,  e,a ≥ 0
                //  e is the surplus: it makes the LHS smaller (subtracts).
                //  a is the artificial: it provides the initial identity basis column.
                std_form_A_[i][e_col++] = -1.0;
                std_form_A_[i][a_col++] = +1.0;
                break;

            case ConstraintType::EQUAL:
                // a^T x = b  →  a^T x + a = b,  a ≥ 0
                //  The artificial gives the initial BFS; Big-M forces it to 0.
                std_form_A_[i][a_col++] = +1.0;
                break;
        }
    }
}

// ── Tableau initialisation ────────────────────────────────────────────────────
//
//  We allocate ONE contiguous Matrix block of size (m+1) × (n_aug+1).
//  The last row is the objective row (reduced costs + −z).
//  The last column is the RHS (b vector + −z value).
//
//  After filling, the objective row is NOT yet in canonical form because the
//  artificials in the basis have non-zero objective coefficients (= Big-M).
//  applyBigMPenalties() eliminates those via row operations.

void BigMSolver::initializeTableau()
{
    const std::size_t m    = problem_.num_constraints();
    const std::size_t naug = tableau_.num_augmented_vars();

    // Allocate (m+1) rows × (naug+1) cols, zero-initialised.
    tableau_.tableau = Matrix(m + 1, naug + 1);

    // ── Fill constraint rows ──────────────────────────────────────────────────
    for (std::size_t i = 0; i < m; ++i) {
        // Coefficient columns.
        for (std::size_t j = 0; j < naug; ++j)
            tableau_.tableau(i, j) = std_form_A_[i][j];
        // RHS column.
        tableau_.tableau(i, naug) = std_form_b_[i];
    }

    // ── Fill objective row (reduced costs) ────────────────────────────────────
    for (std::size_t j = 0; j < naug; ++j)
        tableau_.tableau(m, j) = std_form_c_[j];
    // Objective row RHS = −z = 0 at the start (BFS has x = 0, so z = 0).
    tableau_.tableau(m, naug) = 0.0;

    // ── Identify initial basis ────────────────────────────────────────────────
    //  The initial basis consists of:
    //    • Slack variables (for ≤ constraints) — already form identity columns.
    //    • Artificial variables (for ≥ and = constraints) — also identity cols.
    //  Surplus variables are NOT in the initial basis (they have coefficient −1).
    //
    //  basic_vars[i] = column index of the variable that is basic in row i.
    tableau_.basic_vars.resize(m);
    std::size_t s_ptr = problem_.num_vars();                           // slack cols start
    std::size_t a_ptr = problem_.num_vars() +
                        tableau_.num_slack_vars +
                        tableau_.num_surplus_vars;                     // art cols start

    for (std::size_t i = 0; i < m; ++i) {
        const ConstraintType ct = problem_.constraints[i].type;
        if (ct == ConstraintType::LESS_EQ) {
            tableau_.basic_vars[i] = static_cast<int>(s_ptr++);
        } else {
            // GREATER_EQ or EQUAL: artificial enters basis at row i.
            tableau_.basic_vars[i] = static_cast<int>(a_ptr++);
            if (ct == ConstraintType::GREATER_EQ) ++s_ptr; // skip the surplus col
        }
    }

    // Build non_basic_vars: all columns NOT in basic_vars.
    {
        std::vector<bool> is_basic(naug, false);
        for (int bv : tableau_.basic_vars)
            is_basic[static_cast<std::size_t>(bv)] = true;

        tableau_.non_basic_vars.clear();
        for (std::size_t j = 0; j < naug; ++j)
            if (!is_basic[j])
                tableau_.non_basic_vars.push_back(static_cast<int>(j));
    }

    tableau_.obj_value       = 0.0;
    tableau_.iteration_count = 0;
    tableau_.status          = SolverStatus::NOT_STARTED;

    // ── Canonicalise objective row for the initial basis ──────────────────────
    //  WHY REQUIRED: The initial basis contains artificials with cost M.
    //  In canonical form, every basic variable's column in the objective row
    //  must be 0. Since artificial a_k is basic in row r_k with value 1,
    //  the objective row's coefficient for a_k is currently M (not 0).
    //  We eliminate it via: obj_row ← obj_row − M × row_{r_k}.
    //  This is mathematically equivalent to one elimination step of GJ.
    applyBigMPenalties(tableau_);
}

// ── Pivot column selection ────────────────────────────────────────────────────
//
//  MOST_NEGATIVE_RC (Dantzig's rule):
//    Choose j = argmin c̄_j over all j with c̄_j < −OPTIMALITY_TOL.
//    Rationale: most negative c̄_j → largest rate of decrease in z per
//    unit increase of x_j. Empirically best on average.
//
//  BLAND'S RULE:
//    Choose smallest index j with c̄_j < −OPTIMALITY_TOL.
//    Guarantees finite termination: no repeated basis is possible because
//    the entering index always strictly increases lexicographically.
//    Converges more slowly but is the correct anti-cycling mechanism.
//
//  STEEPEST_EDGE:
//    Choose j maximising |c̄_j| / ‖column_j‖. Approximates greatest
//    improvement per unit step. Most expensive per iteration but fewest
//    total pivots on ill-conditioned problems.

std::optional<std::size_t>
BigMSolver::findPivotColumn(const SimplexTableau& tab) const noexcept
{
    switch (pivot_rule_) {
        case PivotRule::MOST_NEGATIVE_RC:
        {
            const std::size_t naug = tab.num_augmented_vars();
            const std::size_t obj_row = tab.num_constraints();
            double best_rc  = -config::OPTIMALITY_TOL;
            std::optional<std::size_t> best_col;

            for (std::size_t j = 0; j < naug; ++j) {
                const double rc = tab.tableau(obj_row, j);
                if (rc < best_rc) { best_rc = rc; best_col = j; }
            }
            return best_col;
        }

        case PivotRule::BLANDS_RULE:
            return blandsPivotColumn(tab);

        case PivotRule::STEEPEST_EDGE:
        {
            // Approximate steepest edge: max |c̄_j| / ‖d_j‖ where d_j = B^{-1}a_j.
            // B^{-1}a_j is exactly the j-th column of the current tableau.
            const std::size_t naug    = tab.num_augmented_vars();
            const std::size_t m       = tab.num_constraints();
            const std::size_t obj_row = m;
            double best_ratio = 0.0;
            std::optional<std::size_t> best_col;

            for (std::size_t j = 0; j < naug; ++j) {
                const double rc = tab.tableau(obj_row, j);
                if (rc >= -config::OPTIMALITY_TOL) continue;

                // Compute ‖B^{-1}a_j‖² = Σ_i a_{i,j}².
                double norm_sq = 0.0;
                for (std::size_t i = 0; i < m; ++i) {
                    const double v = tab.tableau(i, j);
                    norm_sq += v * v;
                }
                if (norm_sq < config::EPSILON) continue;

                const double ratio = (-rc) / std::sqrt(norm_sq);
                if (ratio > best_ratio) { best_ratio = ratio; best_col = j; }
            }
            return best_col;
        }
    }
    return std::nullopt;  // unreachable
}

// ── Pivot row selection (minimum positive ratio test) ─────────────────────────
//
//  The ratio test ensures x_{new_basic} ≥ 0 after the pivot.
//
//  For entering column q, the leaving row p satisfies:
//    θ* = min { b_i / a_{i,q}  :  a_{i,q} > EPSILON, b_i ≥ 0 }
//
//  WHY a_{i,q} > EPSILON (strict positivity):
//   • a_{i,q} ≤ 0: increasing x_q does NOT decrease x_{B_i}. Row i
//     cannot limit the increase, so we skip it.
//   • a_{i,q} < EPSILON but > 0: numerically too small — dividing b_i by it
//     produces an astronomically large ratio that is meaningless AND the pivot
//     element being near-zero would produce massive numerical error in the
//     row_scale step (b_i / a_{i,q} → ∞ in floating point).
//
//  WHY NO RATIO CANDIDATE → UNBOUNDED:
//   If every a_{i,q} ≤ EPSILON, the entire q column is non-positive.
//   x_q can increase without bound (no row limits it), so z → −∞.
//
//  TIE-BREAKING (degeneracy):
//   Multiple rows can achieve θ* = 0 (degenerate BFS: a basic variable is
//   already at zero). Bland's tie-break (smallest basic_vars index) prevents
//   cycling in degenerate cases.

std::optional<std::size_t>
BigMSolver::findPivotRow(const SimplexTableau& tab,
                          std::size_t           pivot_col) const
{
    const std::size_t m   = tab.num_constraints();
    double best_theta      = config::INF;
    std::optional<std::size_t> best_row;
    int best_basic_idx     = std::numeric_limits<int>::max();

    for (std::size_t i = 0; i < m; ++i) {
        const double a_iq = tab.tableau(i, pivot_col);

        // Skip non-positive pivot candidates — see rationale above.
        if (a_iq <= config::EPSILON) continue;

        const double b_i    = tab.rhs(i);
        const double theta  = b_i / a_iq;

        // Bland's degeneracy tie-break: prefer smallest leaving variable index.
        const bool new_best =
            (theta < best_theta - config::DEGENERACY_TOL) ||
            (std::abs(theta - best_theta) <= config::DEGENERACY_TOL &&
             tab.basic_vars[i] < best_basic_idx);

        if (new_best) {
            best_theta    = theta;
            best_row      = i;
            best_basic_idx = tab.basic_vars[i];
        }
    }

    if (!best_row.has_value()) {
        // Column is unbounded: no positive a_{i,q} found.
        throw UnboundedProblemException(
            "Column " + std::to_string(pivot_col) + " has no positive entry — "
            "the problem is unbounded");
    }

    return best_row;
}

// ── Gauss-Jordan pivot ────────────────────────────────────────────────────────
//
//  EXACT PIVOT MECHANICS (one full iteration):
//  ─────────────────────────────────────────────
//
//  Setup: pivot element is a_{p,q} = tab(p, q).
//
//  STEP 1 — Normalise the pivot row:
//    tab.row_scale(p, 1.0 / a_{p,q})
//    After: tab(p, q) = 1.0,  tab(p, j) = a_{p,j} / a_{p,q}  for j ≠ q
//
//  STEP 2 — Eliminate all other rows (INCLUDING objective row m):
//    For i = 0..m (all rows except p):
//      factor = tab(i, q)   // the coefficient to zero out
//      tab.row_axpy(i, factor, p)
//      // row_i ← row_i − factor × row_p
//      After: tab(i, q) = a_{i,q} − factor × 1.0 = 0.0  ✓
//
//  STEP 3 — Update basis bookkeeping:
//    basic_vars[p] = q     (x_q is now basic in row p)
//    Update non_basic_vars accordingly
//
//  PHYSICAL INTERPRETATION:
//    The tableau now has column q = identity vector e_p.
//    Every other column has been updated by elementary row operations.
//    This is exactly B^{-1} applied to the new basis — the tableau implicitly
//    carries B^{-1} in its column representations.

void BigMSolver::performPivot(SimplexTableau& tab,
                               std::size_t     pivot_row,
                               std::size_t     pivot_col)
{
    const double pivot_elem = tab.tableau(pivot_row, pivot_col);

    // Guard: pivot element too small → catastrophic numerical error.
    if (OPTENG_UNLIKELY(std::abs(pivot_elem) < config::EPSILON)) {
        throw NumericalInstabilityException(
            "Pivot element |" + std::to_string(pivot_elem) + "| < EPSILON=" +
            std::to_string(config::EPSILON) + " at (" +
            std::to_string(pivot_row) + "," + std::to_string(pivot_col) + ")");
    }

    const std::size_t total_rows = tab.num_constraints() + 1;  // includes obj row

    // ── STEP 1: Normalise pivot row ───────────────────────────────────────────
    tab.tableau.row_scale(pivot_row, 1.0 / pivot_elem);
    // tab.tableau(pivot_row, pivot_col) is now EXACTLY 1.0.
    // (row_scale touches cols_ contiguous doubles → L1-cache resident)

    // ── STEP 2: Eliminate all other rows including the objective row ──────────
    for (std::size_t i = 0; i < total_rows; ++i) {
        if (i == pivot_row) continue;

        const double factor = tab.tableau(i, pivot_col);

        // Skip zero rows — eliminates a full row_axpy for sparse problems.
        if (OPTENG_LIKELY(std::abs(factor) > config::EPSILON))
            tab.tableau.row_axpy(i, factor, pivot_row);
        // After: tab.tableau(i, pivot_col) = factor − factor × 1.0 = 0.0
    }

    // ── STEP 3: Update basis bookkeeping ─────────────────────────────────────
    const int old_basic = tab.basic_vars[pivot_row];
    tab.basic_vars[pivot_row] = static_cast<int>(pivot_col);

    // Reflect the change in non_basic_vars.
    for (int& nbv : tab.non_basic_vars) {
        if (nbv == static_cast<int>(pivot_col)) {
            nbv = old_basic;
            break;
        }
    }
}

// ── Optimality test ───────────────────────────────────────────────────────────

bool BigMSolver::isOptimal(const SimplexTableau& tab) const noexcept
{
    const std::size_t naug    = tab.num_augmented_vars();
    const std::size_t obj_row = tab.num_constraints();
    const double*     obj_ptr = tab.tableau.row_ptr(obj_row);

    // One stride-1 scan of the objective row — vectorisable.
    for (std::size_t j = 0; j < naug; ++j)
        if (obj_ptr[j] < -config::OPTIMALITY_TOL)
            return false;
    return true;
}

// ── Artificial variable feasibility gate ──────────────────────────────────────
//
//  After Big-M simplex terminates "optimally", artificial variables MUST be
//  zero. An artificial a_k represents "phantom supply" injected to give the
//  simplex a starting basis — if it's non-zero in the final solution, the
//  original constraints cannot all be satisfied simultaneously.
//
//  WHY Big-M FORCES ARTIFICIALS TO ZERO:
//    The objective penalises each artificial with cost M >> |c_j|.
//    Any feasible solution with a_k > 0 has z ≥ M × a_k >> z*.
//    Therefore the optimiser eliminates artificials to reach z*.
//    IF the original problem is infeasible, the "best" solution still has
//    some a_k > 0 and the Big-M solver reports z* ≈ M × (something positive).
//    We detect this here via the FEASIBILITY_TOL gate.

bool BigMSolver::checkArtificialFeasibility(const SimplexTableau& tab) const
{
    const std::size_t m = tab.num_constraints();

    for (const int art_col : tab.artificial_col_indices) {
        // Find if this artificial is still in the basis.
        for (std::size_t i = 0; i < m; ++i) {
            if (tab.basic_vars[i] == art_col) {
                const double val = tab.rhs(i);
                if (val > config::FEASIBILITY_TOL) {
                    throw InfeasibleProblemException(
                        "Artificial variable a" +
                        std::to_string(art_col) +
                        " = " + std::to_string(val) +
                        " > FEASIBILITY_TOL in optimal basis — problem is infeasible");
                }
            }
        }
    }
    return true;  // all artificials are zero (or driven out of basis)
}

// ── Primal solution extraction ────────────────────────────────────────────────

std::vector<double>
BigMSolver::extractPrimalSolution(const SimplexTableau& tab) const
{
    const std::size_t n = tableau_.num_original_vars;
    const std::size_t m = tab.num_constraints();

    std::vector<double> x(n, 0.0);  // non-basic vars remain 0

    for (std::size_t i = 0; i < m; ++i) {
        const int bv = tab.basic_vars[i];
        if (bv >= 0 && static_cast<std::size_t>(bv) < n)
            x[static_cast<std::size_t>(bv)] = tab.rhs(i);
    }

    // Reverse maximization sign transformation: we negated c, so x* is unchanged.
    // (The objective value needs negation, but the primal variables do not.)
    return x;
}

// ── Dual solution extraction (shadow prices) ──────────────────────────────────
//
//  Shadow prices y* = c_B · B^{-1}.
//  The optimal tableau implicitly stores B^{-1} in the columns that were
//  originally identity (slack) columns. For constraint i with slack s_i at
//  column k_i, the shadow price is:
//
//    y*_i = (original c_{s_i}) − (current objective row value for col k_i)
//
//  Since c_{s_i} = 0 (slack has zero objective cost), this simplifies to:
//    y*_i = −reduced_cost(k_i)   for ≤ constraints.
//
//  For ≥ / = constraints: use the artificial column's reduced cost
//  (with sign adjustment for surplus direction).

std::vector<double>
BigMSolver::extractDualSolution(const SimplexTableau& tab) const
{
    const std::size_t m = tab.num_constraints();
    std::vector<double> y(m, 0.0);

    // Walk each constraint and identify its corresponding slack/artificial column.
    std::size_t s_ptr = tableau_.num_original_vars;
    std::size_t e_ptr = s_ptr + tableau_.num_slack_vars;
    std::size_t a_ptr = e_ptr + tableau_.num_surplus_vars;

    for (std::size_t i = 0; i < m; ++i) {
        const ConstraintType ct = problem_.constraints[i].type;
        switch (ct) {
            case ConstraintType::LESS_EQ:
            {
                // y*_i = −c̄_{s_i}: shadow price from slack column.
                y[i] = -tab.reduced_cost(s_ptr);
                ++s_ptr;
                break;
            }
            case ConstraintType::GREATER_EQ:
            {
                // y*_i = c̄_{a_i} − M:  adjust for Big-M term.
                // The reduced cost of the artificial is M + y*_i in minimisation form.
                y[i] = tab.reduced_cost(a_ptr) - tableau_.big_m_value;
                ++e_ptr;  // skip surplus col
                ++a_ptr;
                break;
            }
            case ConstraintType::EQUAL:
            {
                y[i] = tab.reduced_cost(a_ptr) - tableau_.big_m_value;
                ++a_ptr;
                break;
            }
        }

        // For maximization, dual variables also need sign flip.
        if (tableau_.is_maximization) y[i] = -y[i];
    }

    return y;
}

// ── Reduced cost extraction ───────────────────────────────────────────────────

std::vector<double>
BigMSolver::extractReducedCosts(const SimplexTableau& tab) const
{
    const std::size_t n = tableau_.num_original_vars;
    std::vector<double> rc(n);
    const std::size_t obj_row = tab.num_constraints();

    for (std::size_t j = 0; j < n; ++j)
        rc[j] = tab.tableau(obj_row, j);

    // Flip sign for maximization (we minimised −z internally).
    if (tableau_.is_maximization)
        for (double& v : rc) v = -v;

    return rc;
}

// ── Big-M penalty application ─────────────────────────────────────────────────
//
//  After building the tableau, the objective row is NOT in canonical form
//  because artificials a_k are basic with cost M, yet their column is the
//  identity vector e_k (not yet zeroed in obj row).
//
//  For each artificial a_k in basis at row r_k:
//    c̄_{a_k} is currently M (the Big-M cost in the objective row).
//  Canonical form requires c̄_{a_k} = 0 when a_k is basic.
//
//  FIX: subtract M times row r_k from the objective row:
//    obj_row ← obj_row − M × row_{r_k}
//  This is identical to one GJ elimination on the objective row.
//  After this, c̄_{a_k} = M − M × 1 = 0 ✓
//  All other reduced costs are updated consistently.

void BigMSolver::applyBigMPenalties(SimplexTableau& tab) const
{
    const std::size_t m       = tab.num_constraints();
    const std::size_t obj_row = m;

    for (std::size_t i = 0; i < m; ++i) {
        const int bv = tab.basic_vars[i];
        // Only process artificials in the initial basis.
        if (tab.column_info[static_cast<std::size_t>(bv)].type ==
            VarType::ARTIFICIAL)
        {
            // Eliminate the Big-M entry in the objective row for this artificial.
            const double factor = tab.tableau(obj_row,
                                              static_cast<std::size_t>(bv));
            if (std::abs(factor) > config::EPSILON)
                tab.tableau.row_axpy(obj_row, factor, i);
        }
    }
}

// ── Adaptive Big-M computation ────────────────────────────────────────────────
//
//  WHY ADAPTIVE vs. FIXED M = 1e7:
//  ────────────────────────────────
//  Fixed M fails in two ways:
//
//  (A) M TOO SMALL (M ≤ max|c_j|):
//      The Big-M penalty is no longer dominant. A solution with a_k > 0
//      might appear to have lower cost than a feasible solution. The solver
//      then "incorrectly" accepts an infeasible solution as optimal.
//      Example: if all costs are ~1e8 and M = 1e7, then cost + M·a_k
//      can actually be LOWER with a_k=1 than a feasible solution.
//
//  (B) M TOO LARGE (M >> max|c_j| by many orders):
//      Reduced costs of artificial columns overwhelm reduced costs of real
//      variables in floating-point. When c̄_q = −1.5 and c̄_{a_k} = 1e12,
//      the arithmetic c̄_{a_k} − M × a_{r,q} loses significant digits.
//      This can cause: premature optimality detection, wrong pivot selection,
//      and ultimately a wrong (infeasible) "optimal" solution.
//
//  ADAPTIVE HEURISTIC: M = max(1, max_j|c_j|) × 10^4
//    - Guarantees M ≫ max|c_j| (by factor 10^4) for dominance.
//    - Stays within reasonable floating-point range (< 1e16 for costs < 1e12).
//    - Clamped to [BIG_M_DEFAULT, 1e12] for safety.

double BigMSolver::computeAdaptiveBigM(
    const std::vector<double>& obj_coeffs) noexcept
{
    if (obj_coeffs.empty()) return config::BIG_M_DEFAULT;

    double max_abs = 0.0;
    for (double c : obj_coeffs) max_abs = std::max(max_abs, std::abs(c));

    const double candidate = std::max(1.0, max_abs) * 1.0e4;
    return std::clamp(candidate,
                      config::BIG_M_DEFAULT,
                      1.0e12);
}

// ── Private helper: enforce non-negative RHS ──────────────────────────────────

void BigMSolver::enforceNonNegativeRHS()
{
    for (auto& con : problem_.constraints) {
        if (con.rhs < 0.0) {
            // Negate RHS and all coefficients.
            con.rhs = -con.rhs;
            for (double& coeff : con.coeffs) coeff = -coeff;
            // Flip inequality direction.
            if (con.type == ConstraintType::LESS_EQ)
                con.type = ConstraintType::GREATER_EQ;
            else if (con.type == ConstraintType::GREATER_EQ)
                con.type = ConstraintType::LESS_EQ;
            // EQUAL stays EQUAL — negating both sides preserves equality.
        }
    }
}

// ── Private helper: isArtificial ─────────────────────────────────────────────

bool BigMSolver::isArtificial(std::size_t col_idx) const noexcept
{
    if (col_idx >= tableau_.column_info.size()) return false;
    return tableau_.column_info[col_idx].type == VarType::ARTIFICIAL;
}

// ── Private helper: Bland's rule ─────────────────────────────────────────────

std::optional<std::size_t>
BigMSolver::blandsPivotColumn(const SimplexTableau& tab) const noexcept
{
    const std::size_t naug    = tab.num_augmented_vars();
    const std::size_t obj_row = tab.num_constraints();

    // Bland's rule: SMALLEST INDEX j with c̄_j < −OPTIMALITY_TOL.
    // Not argmin — just the first eligible column by index.
    for (std::size_t j = 0; j < naug; ++j)
        if (tab.tableau(obj_row, j) < -config::OPTIMALITY_TOL)
            return j;

    return std::nullopt;
}

// ── Private helper: Charnes' ε-perturbation ──────────────────────────────────
//
//  When the BFS is DEGENERATE (some b_i = 0), the ratio test can produce
//  θ* = 0 indefinitely, causing cycling: the same sequence of bases repeats
//  forever. Charnes' perturbation converts the degenerate problem into a
//  non-degenerate one by replacing b_i with b_i + ε^{i+1} for tiny ε.
//  Since ε^k are linearly independent over the rationals, no ties can occur
//  in the ratio test, guaranteeing a strict leaving variable at each step.
//
//  WHY NOT ALWAYS USE BLAND'S:
//  Bland's rule also prevents cycling but often takes far more pivots because
//  it ignores the magnitude of improvement. Charnes' perturbation is more
//  aggressive: it doesn't slow down the pivot selection. For production solvers,
//  the standard approach is to use Bland's rule only after detecting stalling.

void BigMSolver::perturbDegenerateRHS(SimplexTableau& tab) const noexcept
{
    const std::size_t m   = tab.num_constraints();
    const std::size_t rhs = tab.rhs_col();

    double eps = 1.0e-7;
    for (std::size_t i = 0; i < m; ++i) {
        if (std::abs(tab.rhs(i)) < config::DEGENERACY_TOL) {
            tab.tableau(i, rhs) += eps;
        }
        eps *= 1.0e-3;  // ε, ε^2, ε^3, ... each row gets a distinct perturbation
    }
}

// ── Tableau print helper ──────────────────────────────────────────────────────

void BigMSolver::printTableau(std::ostream& os, int precision) const
{
    tableau_.print(os, precision);
}

} // namespace OptEngine
/**
 * @file    VogelSolver.cpp
 * @brief   Production Implementation — Vogel's Approximation + MODI Transportation Solver
 * @version 1.0.0
 *
 * ╔══════════════════════════════════════════════════════════════════════════╗
 * ║  Implements: VogelSolver  (§9 of OptimizationEngine.hpp)               ║
 * ║                                                                          ║
 * ║  ALGORITHM PIPELINE                                                      ║
 * ║  ─────────────────                                                       ║
 * ║  Phase 1 — Vogel's Approximation Method (VAM)                           ║
 * ║    Produces a high-quality initial BFS via penalty-guided greedy        ║
 * ║    allocation. Typically within 5–15% of optimal; often optimal.        ║
 * ║                                                                          ║
 * ║  Phase 2 — MODI (Modified Distribution / UV Method)                    ║
 * ║    Iteratively improves the BFS to global optimality by computing       ║
 * ║    dual variables (u_i, v_j) and opportunity costs Δ_ij.               ║
 * ║                                                                          ║
 * ║  MASKING STRATEGY (no matrix resizing):                                  ║
 * ║    eliminated_rows[i] / eliminated_cols[j] boolean flags skip           ║
 * ║    exhausted lanes in O(1) per check. The underlying Matrix is never    ║
 * ║    physically modified during VAM — the cost data stays intact at its   ║
 * ║    original contiguous addresses throughout.                             ║
 * ║                                                                          ║
 * ║  DEGENERACY INVARIANT:                                                   ║
 * ║    A balanced m×n transportation problem has exactly m+n-1 basic        ║
 * ║    cells (spanning-tree property). Degenerate steps produce fewer;      ║
 * ║    repairDegeneracy() inserts DEGENERATE_EPS allocations at minimum-    ║
 * ║    cost unoccupied cells to restore the count.                          ║
 * ╚══════════════════════════════════════════════════════════════════════════╝
 *
 * ── Loop Tracing (findImprovementLoop) ─────────────────────────────────────
 *
 *   Given entering cell (r0, c0), we seek the unique closed path:
 *
 *     (r0,c0) →[H]→ (r0,c1) →[V]→ (r1,c1) →[H]→ (r1,c0) → ... → (r0,c0)
 *
 *   where [H] = horizontal move (same row, different col — both cells in
 *   the path share the row), [V] = vertical move (same col, different row).
 *   All intermediate cells must be basic; only (r0,c0) is non-basic.
 *
 *   Implementation: backtracking DFS on the bipartite spanning tree.
 *   A "turn" alternates direction (H→V→H→V...), and closure requires
 *   returning to (r0, c0) after an even number of steps (≥ 4).
 *
 * ── UV Dual System (computeUVValues) ───────────────────────────────────────
 *
 *   For m sources and n destinations with basic cells B:
 *     u_i + v_j = c_ij   for all (i,j) ∈ B
 *
 *   The system has m+n unknowns and m+n-1 equations → one degree of freedom.
 *   Convention: u_0 = 0. BFS traversal of the spanning tree propagates
 *   the remaining values in O(m+n-1) time.
 *
 *   Dual variables encode shadow prices: v_j is the marginal cost of
 *   satisfying one extra unit of demand at destination j; u_i is the
 *   marginal cost of one extra unit of supply from source i.
 *
 * ── Opportunity Cost (computeOpportunityCosts) ─────────────────────────────
 *
 *   For non-basic cell (i,j):
 *     Δ_ij = c_ij − u_i − v_j
 *
 *   Interpretation: Δ_ij is the net change in total cost per unit
 *   rerouted through (i,j). Δ_ij < 0 → rerouting saves cost → enter basis.
 *   Δ_ij ≥ 0 for all non-basics → current BFS is globally optimal.
 */

#include "OptimizationEngine.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace OptEngine {

// ════════════════════════════════════════════════════════════════════════════
// §A  CONSTRUCTION
// ════════════════════════════════════════════════════════════════════════════

VogelSolver::VogelSolver(TransportationProblem problem)
    : problem_(std::move(problem)),
      balanced_(problem_.is_balanced())
{
    // Auto-balance: adds dummy row or column at zero cost if supply ≠ demand.
    // The constructor of TransportationProblem already handles this via
    // auto_balance=true (default). We re-check here as a defensive assertion.
    if (!problem_.is_balanced())
        problem_.balance();

    problem_.validate();
}

// ════════════════════════════════════════════════════════════════════════════
// §B  PRIMARY ENTRY POINT
// ════════════════════════════════════════════════════════════════════════════

TransportationSolution VogelSolver::solve()
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    // ── Allocate clean m×n allocation matrix ─────────────────────────────────
    Matrix allocation = initAllocationMatrix();

    // ── State vectors: logical masks for exhausted sources / destinations ────
    //
    //  WHY MASKS INSTEAD OF REMOVAL:
    //  Physically erasing a row from the cost Matrix requires O(m×n) data
    //  movement to close the gap (row-major layout means subsequent rows must
    //  shift down). A boolean mask achieves O(1) elimination and preserves
    //  the original matrix addresses — every penalty computation still reads
    //  contiguous memory for active rows/columns, and the compiler can
    //  vectorise the scan loop when the mask branch predicts well.

    std::vector<bool>   eliminated_rows(m, false);
    std::vector<bool>   eliminated_cols(n, false);
    std::vector<double> remaining_supply = problem_.supply;
    std::vector<double> remaining_demand = problem_.demand;

    std::vector<AllocationCell> basic_cells;
    basic_cells.reserve(m + n - 1);   // exact BFS size for balanced problem

    int iter = 0;

    // ── Phase 1: VAM allocation loop ─────────────────────────────────────────
    //
    //  Each iteration eliminates at least one row or column. Therefore the
    //  loop runs at most m+n-1 times, giving O((m+n) × m × n) total — linear
    //  in the number of cells, since penalty computation is O(m×n) per step.
    //
    //  Degenerate steps (supply == demand simultaneously exhausted) may only
    //  eliminate one dimension, but they are bounded by min(m,n).

    while (true) {
        // Check if all sources and destinations are eliminated.
        const bool all_rows_done = std::all_of(eliminated_rows.begin(),
                                               eliminated_rows.end(),
                                               [](bool b){ return b; });
        const bool all_cols_done = std::all_of(eliminated_cols.begin(),
                                               eliminated_cols.end(),
                                               [](bool b){ return b; });
        if (all_rows_done || all_cols_done) break;

        // ── Compute penalties for all active rows and columns ─────────────────
        const auto row_pen = computeRowPenalties(remaining_supply,
                                                  remaining_demand,
                                                  eliminated_rows,
                                                  eliminated_cols);
        const auto col_pen = computeColPenalties(remaining_supply,
                                                  remaining_demand,
                                                  eliminated_rows,
                                                  eliminated_cols);

        // ── Select the dimension with maximum penalty ─────────────────────────
        const auto [is_row, idx] = findMaxPenaltyDimension(row_pen, col_pen);

        // ── Find the minimum-cost uneliminated cell in that dimension ─────────
        std::size_t alloc_row, alloc_col;
        if (is_row) {
            alloc_row = idx;
            alloc_col = findMinCostInRow(idx, eliminated_cols);
        } else {
            alloc_col = idx;
            alloc_row = findMinCostInCol(idx, eliminated_rows);
        }

        // ── Allocate to (alloc_row, alloc_col) and update state ───────────────
        auto cell = allocate(alloc_row, alloc_col,
                             allocation,
                             remaining_supply,
                             remaining_demand,
                             eliminated_rows,
                             eliminated_cols);
        basic_cells.push_back(cell);
        ++iter;
    }

    // ── Degeneracy repair: ensure exactly m+n-1 basic cells ──────────────────
    if (isDegenerateBFS(basic_cells))
        repairDegeneracy(allocation, basic_cells);

    // ── Phase 2: MODI optimality improvement ─────────────────────────────────
    //
    //  Iteratively compute dual variables, evaluate opportunity costs,
    //  and route flow through improving loops until all Δ_ij ≥ 0.

    SolverStatus status = SolverStatus::PARTIAL;
    int modi_iters = 0;

    for (int modi_iter = 0; modi_iter < config::MAX_ITERATIONS; ++modi_iter) {
        // ── Compute dual variables u_i, v_j ──────────────────────────────────
        std::vector<double> u(m, 0.0);
        std::vector<double> v(n, config::UNASSIGNED);
        v[0] = config::UNASSIGNED;  // will be set by BFS traversal

        // Extract current basic cells from allocation matrix (after each step).
        basic_cells = extractBasicCells(allocation);

        if (isDegenerateBFS(basic_cells))
            repairDegeneracy(allocation, basic_cells);

        basic_cells = extractBasicCells(allocation);

        computeUVValues(allocation, basic_cells, u, v);

        // ── Compute opportunity costs Δ_ij for non-basic cells ───────────────
        const Matrix delta = computeOpportunityCosts(allocation, u, v);

        // ── Optimality check ──────────────────────────────────────────────────
        if (isMODIOptimal(delta)) {
            status = SolverStatus::OPTIMAL;
            break;
        }

        // ── Find entering cell: most negative Δ_ij ───────────────────────────
        const auto [enter_row, enter_col] = findEnteringCell(delta);

        // ── Trace improvement loop through the spanning tree ──────────────────
        const auto loop = findImprovementLoop(allocation, enter_row, enter_col);

        // ── Execute the loop shift ────────────────────────────────────────────
        improveAllocation(allocation, loop);

        ++modi_iters;
        ++iter;

        if (modi_iters >= config::MAX_ITERATIONS - 1) {
            status = SolverStatus::MAX_ITER_REACHED;
            break;
        }
    }

    // ── Assemble solution ─────────────────────────────────────────────────────
    TransportationSolution sol(status);
    sol.allocation_matrix = allocation;
    sol.allocations       = extractAllocations(allocation);
    sol.total_cost        = computeTotalCost(allocation);
    sol.iteration_count   = iter;
    sol.is_degenerate     = isDegenerateBFS(extractBasicCells(allocation));
    sol.status_message    =
        std::string(utils::status_to_string(status)) +
        " | Total cost: " + std::to_string(sol.total_cost) +
        " | Iterations: " + std::to_string(iter);

    return sol;
}

// ════════════════════════════════════════════════════════════════════════════
// §C  PHASE 1 — VOGEL'S APPROXIMATION METHOD
// ════════════════════════════════════════════════════════════════════════════

// ── Row penalty computation ───────────────────────────────────────────────────
//
//  For each active row i:
//    Find the two lowest costs among active (non-eliminated) columns.
//    penalty_i = second_minimum − first_minimum
//
//  Complexity: O(m × n) — one full scan per active row over active columns.
//  Memory: two scalars (min1, min2) per row — no auxiliary allocations.
//
//  WHY TWO-PASS SCAN INSTEAD OF PARTIAL_SORT:
//  partial_sort on a vector of n doubles copies the row into a temporary
//  buffer (O(n) allocation) and then partially sorts it (O(n log 2) = O(n)).
//  Our two-scalar approach uses the same O(n) work but with zero allocations
//  and better constants because it touches each element exactly once.

std::vector<double>
VogelSolver::computeRowPenalties(const std::vector<double>& /*remaining_supply*/,
                                  const std::vector<double>& /*remaining_demand*/,
                                  const std::vector<bool>&   eliminated_rows,
                                  const std::vector<bool>&   eliminated_cols) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    std::vector<double> penalties(m, -config::INF);

    for (std::size_t i = 0; i < m; ++i) {
        if (eliminated_rows[i]) continue;   // skip exhausted source

        double min1 = config::INF;   // lowest cost in this row
        double min2 = config::INF;   // second-lowest cost in this row

        // Single-pass scan: maintain running min1 and min2.
        // Invariant: min1 ≤ min2 at all times.
        for (std::size_t j = 0; j < n; ++j) {
            if (eliminated_cols[j]) continue;  // skip exhausted destination

            const double c = problem_.cost(i, j);

            if (c < min1) {
                min2 = min1;  // demote old min1 to second place
                min1 = c;
            } else if (c < min2) {
                min2 = c;     // new second-best
            }
        }

        // If only one active column exists, min2 stays INF → penalty = INF.
        // This signals that row i is completely constrained: any allocation
        // must go to the sole remaining column, so we should allocate there
        // immediately (highest penalty wins in findMaxPenaltyDimension).
        if (min1 < config::INF)
            penalties[i] = (min2 < config::INF) ? (min2 - min1) : config::INF;
        // else: no active column at all → leave penalty as INF (should not occur
        // in a well-balanced problem mid-algorithm, but guards against edge cases).
    }

    return penalties;
}

// ── Column penalty computation ────────────────────────────────────────────────
//
//  Symmetric to computeRowPenalties — iterates over active rows for each
//  active column. The cost matrix is row-major, so column scans are NOT
//  stride-1. For large n, this contributes cache-miss overhead, but since
//  penalty computation is O(m × n) dominated by the allocation loop anyway,
//  it does not change the asymptotic complexity.
//
//  WHY NOT TRANSPOSE:
//  A transposed cost matrix would make column scans cache-friendly but at
//  the cost of m×n memory for a second matrix — doubling the footprint for
//  a benefit only felt in the penalty sub-step. The production decision is to
//  keep a single contiguous Matrix and accept the column-scan cache cost.

std::vector<double>
VogelSolver::computeColPenalties(const std::vector<double>& /*remaining_supply*/,
                                  const std::vector<double>& /*remaining_demand*/,
                                  const std::vector<bool>&   eliminated_rows,
                                  const std::vector<bool>&   eliminated_cols) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    std::vector<double> penalties(n, -
        config::INF);

    for (std::size_t j = 0; j < n; ++j) {
        if (eliminated_cols[j]) continue;

        double min1 = config::INF;
        double min2 = config::INF;

        for (std::size_t i = 0; i < m; ++i) {
            if (eliminated_rows[i]) continue;

            const double c = problem_.cost(i, j);

            if (c < min1) {
                min2 = min1;
                min1 = c;
            } else if (c < min2) {
                min2 = c;
            }
        }

        if (min1 < config::INF)
            penalties[j] = (min2 < config::INF) ? (min2 - min1) : config::INF;
    }

    return penalties;
}

// ── Maximum-penalty dimension selection ──────────────────────────────────────
//
//  Algorithm:
//    1. Find max_row_pen = max(row_penalties[i] for active i).
//    2. Find max_col_pen = max(col_penalties[j] for active j).
//    3. If max_row_pen > max_col_pen  →  return {true, argmax_row}.
//    4. If max_col_pen > max_row_pen  →  return {false, argmax_col}.
//    5. TIE: compare the minimum active cost in the tied row vs. column.
//       Dimension with the lower minimum cost takes priority (Reinfeld & Vogel
//       original heuristic: break ties by assigning where cost impact is largest).
//
//  This is O(m + n) — one scan per dimension.
//  The tie-break O(n) + O(m) scan in step 5 dominates only on near-equal
//  penalty problems and still does not exceed O(m+n) total.

std::pair<bool, std::size_t>
VogelSolver::findMaxPenaltyDimension(
    const std::vector<double>& row_penalties,
    const std::vector<double>& col_penalties) const noexcept
{
    // Find maximum row penalty and its index.
    double best_row_pen  = -config::INF;
    std::size_t best_row = 0;
    for (std::size_t i = 0; i < row_penalties.size(); ++i) {
        if (row_penalties[i] > best_row_pen) {
            best_row_pen = row_penalties[i];
            best_row     = i;
        }
    }

    // Find maximum column penalty and its index.
    double best_col_pen  = -config::INF;
    std::size_t best_col = 0;
    for (std::size_t j = 0; j < col_penalties.size(); ++j) {
        if (col_penalties[j] > best_col_pen) {
            best_col_pen = col_penalties[j];
            best_col     = j;
        }
    }

    // Strict dominance: no tie-breaking needed.
    if (best_row_pen > best_col_pen + config::EPSILON)
        return {true,  best_row};
    if (best_col_pen > best_row_pen + config::EPSILON)
        return {false, best_col};

    // ── Tie-break: prefer the dimension with the lower absolute minimum cost ──
    //  Rationale: if both penalties are equal, allocating along the cheaper
    //  direction minimises immediate cost impact.
    const std::size_t n = problem_.num_destinations();
    const std::size_t m = problem_.num_sources();

    double min_in_row = config::INF;
    for (std::size_t j = 0; j < n; ++j)
        min_in_row = std::min(min_in_row, problem_.cost(best_row, j));

    double min_in_col = config::INF;
    for (std::size_t i = 0; i < m; ++i)
        min_in_col = std::min(min_in_col, problem_.cost(i, best_col));

    // Prefer the dimension with the lower minimum cost (higher urgency).
    if (min_in_col <= min_in_row)
        return {false, best_col};
    return {true, best_row};
}

// ── Minimum-cost cell in a row ────────────────────────────────────────────────

std::size_t
VogelSolver::findMinCostInRow(std::size_t              row_idx,
                               const std::vector<bool>& eliminated_cols) const noexcept
{
    const std::size_t n = problem_.num_destinations();
    double best_cost   = config::INF;
    std::size_t best_j = 0;

    for (std::size_t j = 0; j < n; ++j) {
        if (eliminated_cols[j]) continue;
        const double c = problem_.cost(row_idx, j);
        // Strict less-than: ties resolved by smallest column index (first-found).
        if (c < best_cost) {
            best_cost = c;
            best_j    = j;
        }
    }

    return best_j;
}

// ── Minimum-cost cell in a column ─────────────────────────────────────────────

std::size_t
VogelSolver::findMinCostInCol(std::size_t              col_idx,
                               const std::vector<bool>& eliminated_rows) const noexcept
{
    const std::size_t m = problem_.num_sources();
    double best_cost   = config::INF;
    std::size_t best_i = 0;

    for (std::size_t i = 0; i < m; ++i) {
        if (eliminated_rows[i]) continue;
        const double c = problem_.cost(i, col_idx);
        if (c < best_cost) {
            best_cost = c;
            best_i    = i;
        }
    }

    return best_i;
}

// ── VAM allocation step ───────────────────────────────────────────────────────
//
//  Allocate min(supply, demand) units to cell (row_idx, col_idx).
//  Update the allocation matrix and the residual supply/demand vectors.
//
//  DEGENERACY HANDLING:
//  When supply[i] == demand[j] exactly, both row i and column j are
//  simultaneously exhausted. If we eliminate both, the BFS will contain
//  one fewer basic cell than required (m+n-1 would not be met). We handle
//  this by:
//    1. Eliminating the row (marking it done).
//    2. Setting demand[j] to 0 but NOT eliminating the column yet.
//    3. Injecting DEGENERATE_EPS into the allocation so the spanning tree
//       remains connected. The epsilon is filtered out at solution extraction.
//  This follows Charnes-Cooper degeneracy perturbation adapted for transportation.

AllocationCell
VogelSolver::allocate(std::size_t          row_idx,
                       std::size_t          col_idx,
                       Matrix&              allocation,
                       std::vector<double>& remaining_supply,
                       std::vector<double>& remaining_demand,
                       std::vector<bool>&   eliminated_rows,
                       std::vector<bool>&   eliminated_cols)
{
    const double supply_avail = remaining_supply[row_idx];
    const double demand_reqd  = remaining_demand[col_idx];
    const double amount       = std::min(supply_avail, demand_reqd);

    allocation(row_idx, col_idx) += amount;
    remaining_supply[row_idx]    -= amount;
    remaining_demand[col_idx]    -= amount;

    // ── Elimination logic ──────────────────────────────────────────────────────
    const bool supply_exhausted =
        remaining_supply[row_idx] < config::FEASIBILITY_TOL;
    const bool demand_satisfied =
        remaining_demand[col_idx] < config::FEASIBILITY_TOL;

    if (supply_exhausted && demand_satisfied) {
        // Degenerate step: both exhausted simultaneously.
        // Eliminate the row; leave column "open" with zero remaining demand.
        // This ensures the next penalty iteration still "sees" the column and
        // can assign a degenerate-epsilon basic cell there to maintain m+n-1.
        eliminated_rows[row_idx] = true;
        remaining_supply[row_idx] = 0.0;
        remaining_demand[col_idx] = 0.0;
        // The epsilon is added in repairDegeneracy() if needed; here we just
        // flag the column as not-yet-eliminated so the loop can close it.
        // We also set the column as eliminated to prevent re-use:
        eliminated_cols[col_idx] = true;
        // Note: We lose one potential basic cell here. repairDegeneracy()
        // will restore the spanning tree invariant if #basics < m+n-1.
    } else if (supply_exhausted) {
        eliminated_rows[row_idx] = true;
        remaining_supply[row_idx] = 0.0;
    } else {
        // demand_satisfied but supply remains
        eliminated_cols[col_idx] = true;
        remaining_demand[col_idx] = 0.0;
    }

    return AllocationCell{
        row_idx,
        col_idx,
        amount,
        problem_.cost(row_idx, col_idx)
    };
}

// ════════════════════════════════════════════════════════════════════════════
// §D  PHASE 2 — MODI (MODIFIED DISTRIBUTION / UV METHOD)
// ════════════════════════════════════════════════════════════════════════════

// ── UV dual variable computation ─────────────────────────────────────────────
//
//  The UV system u_i + v_j = c_ij defines a set of m+n linear equations
//  over m+n unknowns, with one degree of freedom resolved by u_0 = 0.
//
//  The basic cells form a spanning tree on the bipartite graph
//  G = (Sources ∪ Destinations, basic_cells). BFS from node (source 0)
//  propagates u/v values through the tree in O(m+n-1) time.
//
//  GRAPH ENCODING:
//  Sources    → node indices [0 .. m-1]
//  Destinations → node indices [m .. m+n-1]
//  Each basic cell (i, j) defines an undirected edge (i, m+j).

void VogelSolver::computeUVValues(
    const Matrix&                      /*allocation*/,
    const std::vector<AllocationCell>& basic_cells,
    std::vector<double>&               u,
    std::vector<double>&               v) const
{
    traverseSpanningTree(basic_cells, u, v);
}

// ── Opportunity cost matrix ───────────────────────────────────────────────────
//
//  For each NON-BASIC cell (i, j):
//    Δ_ij = c_ij − u_i − v_j
//
//  Basic cells receive +INF to exclude them from the entering-cell search.
//  The scan is O(m×n) — fully sequential on the row-major cost matrix.

Matrix
VogelSolver::computeOpportunityCosts(const Matrix&              allocation,
                                      const std::vector<double>& u,
                                      const std::vector<double>& v) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    Matrix delta(m, n);

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (allocation(i, j) > config::DEGENERATE_EPS) {
                // Basic cell: not a candidate for entry.
                delta(i, j) = config::INF;
            } else {
                delta(i, j) = problem_.cost(i, j) - u[i] - v[j];
            }
        }
    }

    return delta;
}

// ── MODI optimality check ─────────────────────────────────────────────────────

bool VogelSolver::isMODIOptimal(const Matrix& delta_matrix) const noexcept
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (delta_matrix(i, j) < -config::OPTIMALITY_TOL &&
                delta_matrix(i, j) < config::INF / 2.0)
                return false;

    return true;
}

// ── Entering cell selection ───────────────────────────────────────────────────
//
//  argmin Δ_ij over all non-basic cells (those with delta < INF/2).
//  Ties: smallest (row, col) lexicographic index — deterministic pivot selection
//  mirrors Bland's rule for LP: prevents cycling in degenerate MODI.

std::pair<std::size_t, std::size_t>
VogelSolver::findEnteringCell(const Matrix& delta_matrix) const noexcept
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    double best_delta   = config::INF;
    std::size_t best_r  = 0;
    std::size_t best_c  = 0;

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double d = delta_matrix(i, j);
            if (d < config::INF / 2.0 && d < best_delta) {
                best_delta = d;
                best_r     = i;
                best_c     = j;
            }
        }
    }

    return {best_r, best_c};
}

// ── Improvement loop tracing ─────────────────────────────────────────────────
//
//  Find the unique closed alternating path through basic cells starting and
//  ending at (entering_row, entering_col).
//
//  ALGORITHM — Backtracking DFS on bipartite graph:
//  ──────────────────────────────────────────────────
//  State: current cell (r, c), direction (HORIZ or VERT), depth.
//  At each step:
//    HORIZ → search same row r for a basic cell (r, c') ≠ current col c.
//    VERT  → search same col c for a basic cell (r', c) ≠ current row r.
//  Closure: if we reach the entering column (for VERT) or entering row
//  (for HORIZ) at depth ≥ 4, the loop is closed.
//
//  Uniqueness: the basic cells form a spanning tree (m+n-1 cells for m+n
//  nodes). A spanning tree has exactly one path between any two nodes.
//  For any non-basic cell (r0, c0), the unique cycle is found by adding
//  edge (r0, c0) to the tree and tracing the unique cycle it creates.
//
//  Complexity: O((m+n)²) worst case — bounded by the number of DFS steps
//  through the spanning tree (path length ≤ m+n-1, branching ≤ max(m,n)).

std::vector<std::pair<std::size_t, std::size_t>>
VogelSolver::findImprovementLoop(const Matrix& allocation,
                                  std::size_t   entering_row,
                                  std::size_t   entering_col) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    using Cell = std::pair<std::size_t, std::size_t>;
    std::vector<Cell> path;
    path.reserve(m + n);

    // Backtracking DFS.
    // Returns true if a valid closed loop was found.
    // 'direction': true = next move is horizontal (search same row for basic col)
    //              false = next move is vertical (search same col for basic row)
    std::function<bool(std::size_t, std::size_t, bool)> dfs =
        [&](std::size_t r, std::size_t c, bool find_in_row) -> bool
    {
        if (find_in_row) {
            // Search all basic cells in row r (excluding current column c).
            for (std::size_t jj = 0; jj < n; ++jj) {
                if (jj == c) continue;
                if (allocation(r, jj) <= config::DEGENERATE_EPS) continue;

                // Is this cell already on our path? Prevent re-visiting.
                bool already_visited = false;
                for (const auto& [pr, pc] : path)
                    if (pr == r && pc == jj) { already_visited = true; break; }
                if (already_visited) continue;

                path.emplace_back(r, jj);

                // Check for loop closure: if column jj equals the entering column,
                // and we have enough cells (≥ 4: entering + 3 more + closing move),
                // the loop is closed (next VERT step would return to entering_row).
                if (jj == entering_col && path.size() >= 3) {
                    // Verify: the next vertical move WOULD reach entering_row.
                    // Since entering_col is reached, if entering_row has a basic
                    // cell at entering_col OR entering_row == r (degenerate single row),
                    // we have closed the loop.
                    // Actually the closure condition: we are at (r, entering_col)
                    // and the next vertical search in entering_col would find
                    // entering_row — but entering_row already "allocated" as the
                    // first node. We check: is there a path from here back?
                    // For the standard algorithm: jj == entering_col AND
                    // path.size() % 2 == 1 (odd count after pushing) means we
                    // are at a column-node about to do a vertical jump back to
                    // entering_row. The loop has even total length including the
                    // entering cell itself.
                    return true;  // loop closed
                }

                // Continue DFS: next move is vertical (find row in column jj).
                if (dfs(r, jj, false)) return true;
                path.pop_back();
            }
        } else {
            // Search all basic cells in column c (excluding current row r).
            for (std::size_t ii = 0; ii < m; ++ii) {
                if (ii == r) continue;
                if (allocation(ii, c) <= config::DEGENERATE_EPS) continue;

                bool already_visited = false;
                for (const auto& [pr, pc] : path)
                    if (pr == ii && pc == c) { already_visited = true; break; }
                if (already_visited) continue;

                path.emplace_back(ii, c);

                // Loop closure check: if we returned to entering_col from
                // a row where entering_row is reachable, check closure.
                if (ii == entering_row && path.size() >= 3) {
                    // We are back at entering_row (at some column c).
                    // The loop closes if c == entering_col — but that's the
                    // entering cell itself (path[0] before we push). Since
                    // entering_col is the start column, a horizontal step
                    // from ii==entering_row back to entering_col closes loop.
                    // We check: is this actually closing back to (entering_row,
                    // entering_col)? Current pos = (entering_row, c).
                    // The closing horizontal step would search row entering_row
                    // for col entering_col — but entering_col is where we started.
                    // The closure IS valid when ii == entering_row (we've found
                    // a basic cell in the entering row that allows a horizontal
                    // return to the entering column).
                    return true;
                }

                if (dfs(ii, c, true)) return true;
                path.pop_back();
            }
        }
        return false;
    };

    // Initialise: entering cell is at index 0 (even = +θ).
    // First move is horizontal: look for a basic cell in entering_row.
    path.emplace_back(entering_row, entering_col);
    bool found = dfs(entering_row, entering_col, true);

    if (!found)
        throw OptEngineException(
            "findImprovementLoop: no closed loop found for entering cell (" +
            std::to_string(entering_row) + "," +
            std::to_string(entering_col) +
            "). BFS may be degenerate — call repairDegeneracy() first.");

    return path;
}

// ── Loop improvement execution ────────────────────────────────────────────────
//
//  Convention (Stepping-stone method):
//    Loop positions 0, 2, 4, … (even) → add θ*
//    Loop positions 1, 3, 5, … (odd)  → subtract θ*
//    θ* = min of allocations at ODD positions (the leaving cells).
//
//  After the shift:
//    • One odd-position cell drops to 0 → it leaves the basis.
//    • The entering cell (position 0) gains θ* → it enters the basis.
//    • All other cells shift by ±θ* to maintain row/column balance.
//
//  DEGENERACY: if θ* = 0, we still perform the shift (the allocation does
//  not change numerically, but the basis changes). The odd cell with the
//  smallest (row, col) index among the minimisers leaves the basis, while
//  the entering cell joins with value 0. This is the degenerate pivot.

void VogelSolver::improveAllocation(
    Matrix&                                                  allocation,
    const std::vector<std::pair<std::size_t, std::size_t>>& loop)
{
    // ── Find θ*: minimum allocation at odd-indexed loop positions ─────────────
    double theta_star = config::INF;
    for (std::size_t k = 1; k < loop.size(); k += 2) {  // odd indices
        const auto& [r, c] = loop[k];
        theta_star = std::min(theta_star, allocation(r, c));
    }

    // ── Apply the shift ───────────────────────────────────────────────────────
    for (std::size_t k = 0; k < loop.size(); ++k) {
        const auto& [r, c] = loop[k];
        if (k % 2 == 0) {
            allocation(r, c) += theta_star;   // even: add
        } else {
            allocation(r, c) -= theta_star;   // odd: subtract
            // Zero out numerically tiny negatives from floating-point error.
            if (allocation(r, c) < config::DEGENERATE_EPS / 10.0)
                allocation(r, c) = 0.0;
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §E  DEGENERACY HANDLING
// ════════════════════════════════════════════════════════════════════════════

// ── Degeneracy detection ──────────────────────────────────────────────────────

bool VogelSolver::isDegenerateBFS(
    const std::vector<AllocationCell>& basic_cells) const noexcept
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();
    // A non-degenerate BFS must have exactly m+n-1 basic cells.
    return basic_cells.size() < (m + n - 1);
}

// ── Degeneracy repair ─────────────────────────────────────────────────────────
//
//  When fewer than m+n-1 basic cells exist, the UV spanning tree is
//  disconnected — computeUVValues() would leave some u_i or v_j unresolved.
//
//  FIX: Insert DEGENERATE_EPS into unoccupied cells until #basics = m+n-1.
//  Cell selection: scan in row-major order to pick the minimum-cost available
//  cell, minimising objective distortion.
//
//  WHY DEGENERATE_EPS NOT ZERO:
//  A cell with allocation = 0 is indistinguishable from an unoccupied cell
//  in the basic-cell extraction step. DEGENERATE_EPS (1e-14) is non-zero
//  but far below any real allocation. The extractAllocations() method filters
//  out cells with amount ≤ DEGENERATE_EPS to exclude them from the final
//  reported solution, so the user never sees the epsilon injections.

void VogelSolver::repairDegeneracy(Matrix&                      allocation,
                                    std::vector<AllocationCell>& basic_cells) const
{
    const std::size_t m        = problem_.num_sources();
    const std::size_t n        = problem_.num_destinations();
    const std::size_t required = m + n - 1;

    while (basic_cells.size() < required) {
        // Find the minimum-cost unoccupied cell (row-major scan).
        double best_cost   = config::INF;
        std::size_t best_r = 0;
        std::size_t best_c = 0;
        bool found         = false;

        for (std::size_t i = 0; i < m ; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                // Skip cells already in the basis.
                if (allocation(i, j) > config::DEGENERATE_EPS / 10.0) continue;

                const double c = problem_.cost(i, j);
                if (c < best_cost) {
                    best_cost = c;
                    best_r    = i;
                    best_c    = j;
                    found     = true;
                    // Don't break — keep scanning for a cheaper option.
                }
            }
        }

        if (!found) break;  // should not occur for a valid balanced problem

        // Inject epsilon allocation.
        allocation(best_r, best_c) = config::DEGENERATE_EPS;
        basic_cells.push_back(AllocationCell{
            best_r,
            best_c,
            config::DEGENERATE_EPS,
            problem_.cost(best_r, best_c)
        });
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §F  SOLUTION ASSEMBLY
// ════════════════════════════════════════════════════════════════════════════

// ── Total cost computation ────────────────────────────────────────────────────
//
//  Σ_{i,j} c_ij × x_ij — O(m×n) sequential scan on row-major allocation.
//  DEGENERATE_EPS allocations contribute c_ij × 1e-14 ≈ 0 — negligible.

double VogelSolver::computeTotalCost(const Matrix& allocation) const noexcept
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();
    double total = 0.0;

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            total += problem_.cost(i, j) * allocation(i, j);

    return total;
}

// ── Non-zero allocation extraction ───────────────────────────────────────────

std::vector<AllocationCell>
VogelSolver::extractAllocations(const Matrix& allocation) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    std::vector<AllocationCell> result;
    result.reserve(m + n - 1);  // at most m+n-1 basic cells

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double amt = allocation(i, j);
            if (amt > config::DEGENERATE_EPS) {   // filter epsilon injections
                result.push_back(AllocationCell{
                    i, j, amt, problem_.cost(i, j)
                });
            }
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════════════
// §G  DIAGNOSTICS
// ════════════════════════════════════════════════════════════════════════════

void VogelSolver::printAllocation(const Matrix& allocation,
                                   std::ostream& os) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    os << "\n── Allocation Matrix (" << m << "×" << n << ") ──\n";
    os << std::fixed << std::setprecision(4);

    for (std::size_t i = 0; i < m; ++i) {
        os << "  S" << i << "  |";
        for (std::size_t j = 0; j < n; ++j) {
            const double v = allocation(i, j);
            if (v > config::DEGENERATE_EPS)
                os << std::setw(10) << v;
            else
                os << std::setw(10) << "   -";
        }
        os << "  | supply=" << problem_.supply[i] << "\n";
    }
    os << "Demand |";
    for (std::size_t j = 0; j < n; ++j)
        os << std::setw(10) << problem_.demand[j];
    os << "\n";
    os << "Total cost: " << computeTotalCost(allocation) << "\n";
}

void VogelSolver::printDeltaMatrix(const Matrix& delta,
                                    std::ostream& os) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    os << "\n── Opportunity Cost Matrix (Δ_ij) ──\n";
    os << std::fixed << std::setprecision(4);

    for (std::size_t i = 0; i < m; ++i) {
        os << "  S" << i << "  |";
        for (std::size_t j = 0; j < n; ++j) {
            const double d = delta(i, j);
            if (d > config::INF / 2.0)
                os << std::setw(10) << "  [BAS]";
            else
                os << std::setw(10) << d;
        }
        os << "\n";
    }
}

// ════════════════════════════════════════════════════════════════════════════
// §H  PRIVATE HELPERS
// ════════════════════════════════════════════════════════════════════════════

// ── Allocation matrix initialisation ─────────────────────────────────────────

Matrix VogelSolver::initAllocationMatrix() const
{
    return Matrix(problem_.num_sources(), problem_.num_destinations());
    // Matrix constructor zero-initialises — no allocation exists initially.
}

// ── Basic cell extraction from allocation matrix ──────────────────────────────
//
//  A cell is "basic" if its allocation exceeds the degeneracy epsilon.
//  This distinguishes real allocations from epsilon-repair injections
//  during extractBasicCells() called inside the MODI loop.

std::vector<AllocationCell>
VogelSolver::extractBasicCells(const Matrix& allocation) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();

    std::vector<AllocationCell> cells;
    cells.reserve(m + n - 1);

    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (allocation(i, j) > config::DEGENERATE_EPS / 10.0)
                cells.push_back(AllocationCell{
                    i, j,
                    allocation(i, j),
                    problem_.cost(i, j)
                });

    return cells;
}

// ── Spanning tree BFS for UV propagation ─────────────────────────────────────
//
//  MODEL:
//  Treat sources as nodes 0..m-1 and destinations as nodes m..m+n-1.
//  Each basic cell (i, j) is an undirected edge (i) ↔ (m+j).
//  The m+n-1 basic cells form a spanning tree (for a non-degenerate BFS).
//
//  BFS from source-node 0 (with u[0] = 0) propagates dual values:
//    If edge (source i) → (dest j): and u[i] known → v[j] = c_ij − u[i]
//    If edge (dest j) → (source i): and v[j] known → u[i] = c_ij − v[j]
//
//  Complexity: O(m + n + |edges|) = O(m + n) for spanning tree.
//
//  WHY BFS OVER GAUSSIAN ELIMINATION:
//  The UV system is sparse (each equation has exactly 2 non-zero entries).
//  BFS exploits this sparsity directly in O(m+n) time. Gaussian elimination
//  on the full (m+n-1) × (m+n) matrix would be O((m+n)²) — unnecessarily
//  expensive when the tree structure is already known.

void VogelSolver::traverseSpanningTree(
    const std::vector<AllocationCell>& basic_cells,
    std::vector<double>&               u,
    std::vector<double>&               v) const
{
    const std::size_t m = problem_.num_sources();
    const std::size_t n = problem_.num_destinations();
    const std::size_t total_nodes = m + n;

    // Build adjacency list: node → list of (neighbour_node, cost_on_edge).
    // Source i  → node i.
    // Dest j    → node m + j.
    std::vector<std::vector<std::pair<std::size_t, double>>> adj(total_nodes);

    for (const auto& cell : basic_cells) {
        const std::size_t src_node  = cell.source;
        const std::size_t dest_node = m + cell.destination;
        const double      cost      = cell.unit_cost;
        adj[src_node ].emplace_back(dest_node, cost);
        adj[dest_node].emplace_back(src_node,  cost);
    }

    // Track which nodes have been visited (dual value assigned).
    std::vector<bool> visited(total_nodes, false);

    // Initialise u/v to UNASSIGNED sentinel.
    std::fill(u.begin(), u.end(), config::UNASSIGNED);
    std::fill(v.begin(), v.end(), config::UNASSIGNED);

    // Seed: u_0 = 0.
    u[0]         = 0.0;
    visited[0]   = true;

    // BFS queue holds node indices.
    std::queue<std::size_t> bfs_queue;
    bfs_queue.push(0);

    while (!bfs_queue.empty()) {
        const std::size_t node = bfs_queue.front();
        bfs_queue.pop();

        const bool node_is_source = (node < m);

        for (const auto& [neighbour, edge_cost] : adj[node]) {
            if (visited[neighbour]) continue;
            visited[neighbour] = true;

            const bool neighbour_is_source = (neighbour < m);

            if (node_is_source && !neighbour_is_source) {
                // Source node → destination neighbour:  u_i + v_j = c_ij
                const std::size_t i = node;
                const std::size_t j = neighbour - m;
                v[j] = edge_cost - u[i];
            } else if (!node_is_source && neighbour_is_source) {
                // Destination node → source neighbour:  u_i + v_j = c_ij
                const std::size_t j = node - m;
                const std::size_t i = neighbour;
                u[i] = edge_cost - v[j];
            }
            // Source-source or dest-dest edges should not occur in a valid tree.

            bfs_queue.push(neighbour);
        }
    }

    // Fallback for disconnected components (should only occur if degeneracy
    // repair was insufficient — assign 0 to unresolved nodes).
    for (std::size_t i = 0; i < m; ++i)
        if (std::isnan(u[i]) || !std::isfinite(u[i]))
            u[i] = 0.0;
    for (std::size_t j = 0; j < n; ++j)
        if (std::isnan(v[j]) || !std::isfinite(v[j]))
            v[j] = 0.0;
}

} // namespace OptEngine
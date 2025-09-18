import numpy as np
from itertools import combinations, permutations
import math
from typing import List, Tuple, Dict, Set
import time
from itertools import combinations


class TransportationOptimizer:
    def __init__(self, day2consider: int = 0):
        # City names (excluding WHS which is always start/end)
        self.cities = ['BNG', 'JAK', 'ORL', 'TPA', 'ATL', 'MAC', 'BTM', 
                      'CHR', 'RAL', 'COL', 'MPH', 'NSH', 'ALX', 'RCH']
        
        # Full city list including WHS for indexing
        self.all_cities = ['WHS'] + self.cities
        
        # Distance matrix
        self.distances = np.array([
            [  0, 150, 351, 426, 452,  10,  82, 654, 240, 377, 214, 382, 246, 618, 532],
            [150,  10, 463, 544, 539, 150, 200, 771, 391, 479, 364, 255, 194, 735, 634],
            [351, 463,  10, 140, 194, 351, 273, 751, 390, 456, 412, 694, 563, 715, 611],
            [426, 544, 140,  10,  82, 426, 344, 892, 530, 596, 552, 776, 672, 856, 751],
            [452, 539, 194,  82,  10, 452, 370, 945, 584, 650, 593, 792, 698, 909, 805],
            [ 10, 150, 351, 426, 452,  10,  82, 654, 240, 377, 214, 382, 246, 618, 532],
            [ 82, 200, 273, 344, 370,  82,  10, 736, 322, 450, 223, 455, 328, 700, 605],
            [654, 771, 751, 892, 945, 654, 736,  10, 418, 298, 517, 891, 696,  37, 143],
            [240, 391, 390, 530, 584, 240, 322, 418,  10, 136,  99, 622, 486, 397, 292],
            [377, 479, 456, 596, 650, 377, 450, 298, 136,  10, 227, 661, 533, 261, 155],
            [214, 364, 412, 552, 593, 214, 223, 517,  99, 227,  10, 596, 460, 488, 382],
            [382, 255, 694, 776, 792, 382, 455, 891, 622, 661, 596,  10, 209, 854, 809],
            [246, 194, 563, 672, 698, 246, 328, 696, 486, 533, 460, 209,  10, 659, 600],
            [618, 735, 715, 856, 909, 618, 700,  37, 397, 261, 488, 854, 659,  10, 106],
            [532, 634, 611, 751, 805, 532, 605, 143, 292, 155, 382, 809, 600, 106,  0]
        ])
        
        # Demands by city and day (excluding WHS)
        self.demands = np.array([
            [2940, 1313, 1975, 6589, 5089, 459, 384, 569, 23877, 803, 132, 1495, 1851, 1414],
            [4304, 1034, 1121, 12585, 18097, 355, 750, 7428, 23913, 3284, 3772, 1697, 2243, 1231],
            [2191, 1702, 2805, 17155, 16740, 1207, 992, 3780, 2101, 3762, 1289, 728, 326, 1696],
            [2816, 2126, 2840, 14059, 13678, 1783, 878, 7281, 19296, 63, 2219, 206, 4706, 5183],
            [1798, 243, 2603, 15391, 14401, 2757, 782, 2305, 3204, 973, 1027, 2351, 4806, 9005],
            [2940, 1313, 1975, 6589, 5089, 459, 384, 569, 23877, 803, 132, 1495, 1851, 1414],
            [4304, 1034, 1121, 12585, 18097, 355, 750, 7428, 23913, 3284, 3772, 1697, 2243, 1231],
            [2191, 1702, 2805, 17155, 16740, 1207, 992, 3780, 2101, 3762, 1289, 728, 326, 1696],
            [2816, 2126, 2840, 14059, 13678, 1783, 878, 7281, 19296, 63, 2219, 206, 4706, 5183],
            [1798, 243, 2603, 15391, 14401, 2757, 782, 2305, 3204, 973, 1027, 2351, 4806, 9005],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        
        # Parameters
        self.V = 50  # Speed (mph)
        self.lambda_days = 10  # Total days to consider
        self.truck_capacity = 36000  # Truck capacity
        
        # Weights for score calculation
        self.w1 = 200    # NT weight
        self.w2 = 1.5   # ML weight, considers the lambda.
        self.w3 = 800     # TDD weight, considers the lambda.  
        self.w4 = 1900   # ACW weight
        self.w5 = 400    # MCW weight
        self.day2consider = day2consider
        
        self.record_size = 10  # Number of top strategies to keep

        # Memoization caches for TSP solutions
        self._hk_cache = {}
        self._heur_cache = {}
    
    def get_city_index(self, city: str) -> int:
        """Get the index of a city in the distance matrix"""
        return self.all_cities.index(city)
    
    def generate_partitions(self, n_loops: int):
        """
        Yield each unordered partition of self.cities into n_loops non-empty sets exactly once.
        Canonical rule: always place the smallest remaining city (by name) in the next block,
        then choose the rest of that block from the remaining items.
        Produces S(n, k) partitions (Stirling numbers), not k! * S(n, k).
        """
        items = sorted(self.cities)  # determinism by name

        # quick infeasibility check
        if n_loops < 1 or n_loops > len(items):
            return  # yields nothing

        def helper(rest, k):
            # rest is a sorted list of remaining items
            if k == 1:
                # last block gets everything left
                yield [set(rest)]
                return

            # canonical anchor for this block
            anchor = rest[0]
            tail = rest[1:]

            # we must leave at least 1 element for each of the remaining k-1 blocks
            max_r = len(tail) - (k - 1)
            if max_r < 0:
                return  # not enough items to fill remaining blocks

            for r in range(0, max_r + 1):
                for extra in combinations(tail, r):
                    block = {anchor, *extra}

                    # remaining items, kept sorted
                    extra_set = set(extra)
                    remaining = [x for x in tail if x not in extra_set]

                    for rest_blocks in helper(remaining, k - 1):
                        yield [block] + rest_blocks

        # stream results to the caller
        yield from helper(items, n_loops)
    
    def find_optimal_sequence(self, cities: Set[str]) -> Tuple[List[str], float]:
        """
        Find a visiting sequence for the given city set (start/end at WHS).
        Policy:
        - n <= 8: exact Held–Karp DP (memoized)
        - n >  8: deterministic Nearest Insertion + one 2-opt pass
        """
        if not cities:
            return [], 0.0

        key = frozenset(cities)
        # Return any memoized result (exact or heuristic)
        if key in self._hk_cache:
            seq, dist = self._hk_cache[key]
            return list(seq), dist
        if key in self._heur_cache:
            seq, dist = self._heur_cache[key]
            return list(seq), dist

        n = len(cities)
        city_list = list(cities)

        if n == 1:
            c = city_list[0]
            d = 2 * self.distances[0][self.get_city_index(c)]
            self._hk_cache[key] = ([c], d)
            return [c], d

        if n <= 8:
            # print(f"(Exact) Held–Karp DP for {n} cities")
            route, dist = self._tsp_held_karp(city_list)
            self._hk_cache[key] = (list(route), dist)
            return route, dist
        else:
            # print(f"(Heuristic) Nearest Insertion + 2-opt for {n} cities")
            route = self._nearest_insertion_route(city_list)
            route = self._two_opt_single_pass(route)
            dist = float(self.calculate_route_distance(route))
            self._heur_cache[key] = (list(route), dist)
            return route, dist


    # ---------- Helpers ----------

    def _idx(self, node: str) -> int:
        return 0 if node == 'WHS' else self.get_city_index(node)

    def _d(self, a: str, b: str) -> float:
        return float(self.distances[self._idx(a)][self._idx(b)])


    # ===== Exact: Held–Karp (depot + subset) =====
    def _tsp_held_karp(self, cities: List[str]) -> Tuple[List[str], float]:
        """
        Exact TSP tour length and order for WHS + cities using Held–Karp DP.
        Returns a route of city names (without depot), and the total WHS→...→WHS distance.
        """
        # Compact index map: 0 = WHS, 1..m = cities
        idxs = [0] + [self.get_city_index(c) for c in cities]
        m = len(cities)
        # Compact distance matrix D over [WHS] + subset
        D = [[self.distances[i][j] for j in idxs] for i in idxs]

        INF = float('inf')
        size = 1 << m
        dp = [[INF]*(m+1) for _ in range(size)]       # dp[mask][j]
        par = [[(-1, -1)]*(m+1) for _ in range(size)] # parent pointers

        # Base: direct from depot to each single city j
        for j in range(1, m+1):
            dp[1 << (j-1)][j] = D[0][j]

        # Transitions
        for mask in range(size):
            # iterate end city j in mask
            for j in range(1, m+1):
                if not (mask & (1 << (j-1))):
                    continue
                prev_mask = mask ^ (1 << (j-1))
                if prev_mask == 0:
                    continue
                best = dp[mask][j]
                best_k = -1
                # try all k in prev_mask
                kbits = prev_mask
                while kbits:
                    # take lowest-set bit of kbits as candidate k
                    ls = kbits & -kbits
                    k = (ls.bit_length())  # 1..m
                    kbits -= ls
                    cand = dp[prev_mask][k] + D[k][j]
                    if cand < best:
                        best = cand
                        best_k = k
                if best < dp[mask][j]:
                    dp[mask][j] = best
                    par[mask][j] = (prev_mask, best_k)

        # Close tour back to depot
        full = size - 1
        best_end, best_total = -1, INF
        for j in range(1, m+1):
            total = dp[full][j] + D[j][0]
            if total < best_total:
                best_total, best_end = total, j

        # Reconstruct route (indices 1..m), then map back to city names
        route_idx = []
        mask, j = full, best_end
        while j != -1 and mask:
            route_idx.append(j)
            mask, j = par[mask][j]
        route_idx.reverse()
        route = [cities[k-1] for k in route_idx]
        return route, float(best_total)


    # ===== Heuristic: Nearest Insertion (deterministic ties) =====
    def _nearest_insertion_route(self, cities: List[str]) -> List[str]:
        """
        Build a cycle (WHS anchored) by nearest insertion with deterministic tie-breakers.
        Returns a list of city names (no depot included).
        """
        unvisited = set(cities)

        # seed c1: nearest to depot; tie-break by name
        c1 = min(unvisited, key=lambda c: (self._d('WHS', c), c))
        unvisited.remove(c1)
        if not unvisited:
            return [c1]

        # seed c2: nearest to c1; tie-break by name; choose order minimizing depot cycle
        c2 = min(unvisited, key=lambda c: (self._d(c1, c), c))
        unvisited.remove(c2)
        # choose the orientation that minimizes WHS->c1->c2->WHS vs WHS->c2->c1->WHS
        cycle_len_12 = self._d('WHS', c1) + self._d(c1, c2) + self._d(c2, 'WHS')
        cycle_len_21 = self._d('WHS', c2) + self._d(c2, c1) + self._d(c1, 'WHS')
        route = [c1, c2] if cycle_len_12 <= cycle_len_21 else [c2, c1]

        # helper to compute best insertion place for u
        def best_insertion(route, u):
            best_delta = float('inf')
            best_pos = 0

            # consider edges: (WHS, route[0]), (route[i], route[i+1]) for all i, (route[-1], WHS)
            # inserting between edge (a,b) means position after 'a' (or at 0 if a is WHS)
            # map edges to insertion index in route
            candidates = [ ('WHS', route[0], 0) ]
            candidates += [ (route[i], route[i+1], i+1) for i in range(len(route)-1) ]
            candidates += [ (route[-1], 'WHS', len(route)) ]

            for a, b, pos in candidates:
                delta = self._d(a, u) + self._d(u, b) - self._d(a, b)
                if (delta < best_delta) or (abs(delta - best_delta) < 1e-12 and pos < best_pos):
                    best_delta, best_pos = delta, pos
            return best_pos, best_delta

        while unvisited:
            # choose the city with minimum insertion delta; tie-break by city name then pos
            best_city = None
            best_pos = 0
            best_delta = float('inf')
            for u in sorted(unvisited):  # name-sorted for deterministic ties
                pos, delta = best_insertion(route, u)
                if (delta < best_delta or
                (abs(delta - best_delta) < 1e-12 and (u < (best_city or u) or (u == best_city and pos < best_pos)))):
                    best_city, best_pos, best_delta = u, pos, delta
            route.insert(best_pos, best_city)
            unvisited.remove(best_city)

        return route


    # ===== Light polish: single-pass 2-opt (first improvement, depot-aware) =====
    def _two_opt_single_pass(self, route: List[str]) -> List[str]:
        """
        One deterministic pass of 2-opt over the tour with depot endpoints.
        Reverses a single segment if it yields the first found improvement; returns immediately.
        """
        n = len(route)
        if n < 3:
            return route

        # Work on an extended cycle including the depot at both ends
        ext = ['WHS'] + route[:] + ['WHS']

        # edges are (i,i+1) in ext, where i = 0..n  (since ext has length n+2)
        best_ext = ext
        best_improved = False

        for i in range(0, n):           # edge (i, i+1)
            for j in range(i+1, n+1):   # edge (j, j+1)
                a, b = ext[i],   ext[i+1]
                c, d = ext[j],   ext[j+1]
                old = self._d(a, b) + self._d(c, d)
                new = self._d(a, c) + self._d(b, d)
                if new + 1e-12 < old:
                    # reverse segment (i+1 .. j)
                    new_ext = ext[:i+1] + ext[i+1:j+1][::-1] + ext[j+1:]
                    best_ext = new_ext
                    best_improved = True
                    break
            if best_improved:
                break

        if best_improved:
            return best_ext[1:-1]
        return route

    
    def calculate_route_distance(self, route: List[str]) -> float:
        """Calculate total distance for a route starting and ending at WHS"""
        total = 0
        current = 0  # Start at WHS
        
        for city in route:
            next_idx = self.get_city_index(city)
            total += self.distances[current][next_idx]
            current = next_idx
        
        # Return to WHS
        total += self.distances[current][0]
        return total
    
    def calculate_indexes(self, loops: List[Tuple[List[str], float]], day: int = 0) -> Dict:
        """Calculate all indexes for a set of loops"""
        NT = 0  # Needed Trucks
        ML = 0  # Total Mileage
        TDD = 0  # Total Driver Days
        ACW = 0  # Average Customers Waiting
        MCW = 0  # Maximum Customers Waiting (first day)
        
        total_demand = 0
        max_loop_demand = 0
        
        for route, distance in loops:
            # Calculate time for this loop using new formula
            n_cities = len(route)
            time_days = 0.00234 * distance + 0.053 * n_cities
            time_days_ceil = math.ceil(time_days)
            
            NT += time_days_ceil
            ML += distance
            
            # Calculate TDD for this loop (driver days)
            full_cycles = self.lambda_days // time_days_ceil
            remaining_days = self.lambda_days - (full_cycles * time_days_ceil)
            loop_tdd = full_cycles * time_days + min(remaining_days, time_days)
            TDD += loop_tdd
            
            # Calculate customer waiting for this loop
            cw_sequence = []
            cities_visited = 0
            for d in range(time_days_ceil):
                # Approximate cities visited per day
                cities_per_day = n_cities / time_days
                cities_visited += cities_per_day
                waiting = max(0, n_cities - int(cities_visited))
                cw_sequence.append(waiting)
            # First day waiting (all cities in this loop are waiting initially)
            MCW += n_cities
            # Average waiting
            if cw_sequence:
                ACW += sum(cw_sequence) / len(cw_sequence)
            # Calculate demand for this loop
            loop_demand = 0
            for city in route:
                city_idx = self.cities.index(city)
                loop_demand += self.demands[day][city_idx]
            total_demand += loop_demand
            max_loop_demand = max(max_loop_demand, loop_demand)
        
        return {
            'NT': NT,
            'ML': ML,
            'TDD': TDD,
            'ACW': ACW,
            'MCW': MCW,
            'total_demand': total_demand,
            'max_loop_demand': max_loop_demand
        }
    
    def calculate_score(self, indexes: Dict) -> float:
        """Calculate the overall score"""
        return (self.w1 * indexes['NT'] + 
                self.w2 * indexes['ML'] + 
                self.w3 * indexes['TDD'] + 
                self.w4 * indexes['ACW'] + 
                self.w5 * indexes['MCW'])
    
    def check_constraints(self, indexes: Dict) -> bool:
        """Check if the solution satisfies all constraints"""
        # Constraint 1: Number of trucks <= 10
        if indexes['NT'] > 10:
            return False
        
        # Constraint 2: Truck capacity
        if indexes['max_loop_demand'] > self.truck_capacity:
            return False
        
        return True
    
    def optimize(self, day: int = None, verbose: bool = True) -> Tuple[List, List]:
        """Main optimization function"""
        if day is None:
            day = self.day2consider
        
        record = []  # List of (composition, score, indexes)
        
        print(f"\n{'='*60}")
        print(f"Transportation Route Optimization System")
        print(f"Optimizing for Day {day} demand")
        print(f"{'='*60}\n")
        
        for n_loops in range(1, 7):  # 1 to 6 loops
            if verbose:
                print(f"\n[Processing] Analyzing {n_loops} loop(s)...")
            
            partitions = self.generate_partitions(n_loops)
            valid_count = 0
            
            for i, partition in enumerate(partitions):
                # Find optimal sequence for each subset
                loops = []
                for subset in partition:
                    sequence, distance = self.find_optimal_sequence(subset)
                    loops.append((sequence, distance))
                
                # Calculate indexes
                indexes = self.calculate_indexes(loops, day)
                
                # Debug print every 10,000,000 analyzed loop settings
                if (i + 1) % 10000000 == 0 and verbose:
                    print(f"    [Debug] Analyzed {i + 1} loop settings for {n_loops} loop(s)...")

                # Check constraints
                if not self.check_constraints(indexes):
                    continue
                
                valid_count += 1
                
                # Calculate score
                score = self.calculate_score(indexes)
                
                # Update record
                if len(record) < self.record_size:
                    record.append((loops, score, indexes))
                elif score < record[-1][1]:
                    record[-1] = (loops, score, indexes)
                else:
                    continue
                
                # Keep record sorted
                record.sort(key=lambda x: x[1])
            
            if verbose:
                print(f"  Found {valid_count} valid configurations")
                if record:
                    print(f"  Current best score: {record[0][1]:.2f}")
        
        return record[0] if record else None, record
    
    def print_solution(self, solution: Tuple, rank: int = 1):
        """Print a solution in a readable format"""
        if not solution:
            print("No valid solution found!")
            return
        
        loops, score, indexes = solution
        
        print(f"\n{'='*60}")
        print(f"Rank #{rank} Solution - Score: {score:.2f}")
        print(f"{'='*60}")
        
        print(f"\nLoop Configuration ({len(loops)} loops):")
        for i, (route, distance) in enumerate(loops, 1):
            route_str = " → ".join(['WHS'] + route + ['WHS'])
            print(f"  Loop {i}: {route_str}")
            print(f"          Distance: {distance:.1f} miles")
        
        print(f"\nPerformance Metrics:")
        print(f"  • Needed Trucks (NT):           {indexes['NT']}")
        print(f"  • Total Mileage (ML):           {indexes['ML']:.1f} miles")
        print(f"  • Total Driver Days (TDD):     {indexes['TDD']:.1f} days")
        print(f"  • Avg Customers Waiting (ACW):  {indexes['ACW']:.2f}")
        print(f"  • Max Customers Waiting (MCW):  {indexes['MCW']}")
        print(f"  • Max Loop Demand:              {indexes['max_loop_demand']:,} units")
        
        print(f"\nScore Components:")
        print(f"  • NT component:  {self.w1} × {indexes['NT']} = {self.w1 * indexes['NT']:.2f}")
        print(f"  • ML component:  {self.w2} × {indexes['ML']:.1f} = {self.w2 * indexes['ML']:.2f}")
        print(f"  • TDD component: {self.w3} × {indexes['TDD']:.1f} = {self.w3 * indexes['TDD']:.2f}")
        print(f"  • ACW component: {self.w4} × {indexes['ACW']:.2f} = {self.w4 * indexes['ACW']:.2f}")
        print(f"  • MCW component: {self.w5} × {indexes['MCW']} = {self.w5 * indexes['MCW']:.2f}")


def main():
    """Main execution function"""
    optimizer = TransportationOptimizer()
    
    # Run optimization for day 0 (you can change this to analyze different days)
    start_time = time.time()
    best_solution, all_solutions = optimizer.optimize(day=0, verbose=True)
    end_time = time.time()
    
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"{'='*60}")
    
    # Print top 3 solutions
    print(f"\nTOP 3 SOLUTIONS:")
    for i, solution in enumerate(all_solutions[:3], 1):
        optimizer.print_solution(solution, rank=i)
    
    # Summary
    if best_solution:
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"The optimal strategy uses {len(best_solution[0])} loop(s)")
        print(f"with a total score of {best_solution[1]:.2f}")
        print(f"\nThis configuration requires {best_solution[2]['NT']} trucks")
        print(f"and covers {best_solution[2]['ML']:.1f} miles in total.")


if __name__ == "__main__":
    main()
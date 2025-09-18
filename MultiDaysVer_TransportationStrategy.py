import numpy as np
from itertools import combinations, permutations, product
import math
from typing import List, Tuple, Dict, Set, Optional
import time
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class Loop:
    """Represents a delivery loop"""
    route: List[str]
    distance: float
    duration_days: int  # ceil(time_days)
    time_days: float   # actual time in days
    cities: Set[str]
    
    def __repr__(self):
        return f"Loop(cities={len(self.cities)}, duration={self.duration_days}d, distance={self.distance:.1f}mi)"


@dataclass
class DayStrategy:
    """Strategy for a single day"""
    loops: List[Loop]
    score: float
    indexes: Dict
    day_type: int  # 0=Monday, 1=Tuesday, etc.
    
    def __repr__(self):
        return f"Strategy(loops={len(self.loops)}, score={self.score:.2f})"


@dataclass
class TruckState:
    """Tracks the state of a single truck"""
    id: int
    busy_until_day: int  # -1 if available
    current_loop: Optional[Loop] = None
    
    def is_available(self, day: int) -> bool:
        return self.busy_until_day < day


class SingleDayOptimizer:
    """Encapsulated single-day optimization logic"""
    
    def __init__(self, distances: np.ndarray, demands: np.ndarray, 
                 cities: List[str], all_cities: List[str]):
        self.distances = distances
        self.demands = demands
        self.cities = cities
        self.all_cities = all_cities
        
        # Parameters
        self.V = 50  # Speed (mph)
        self.truck_capacity = 36000
        
        # Caches for TSP solutions
        self._hk_cache = {}
        self._heur_cache = {}
    
    def get_city_index(self, city: str) -> int:
        """Get the index of a city in the distance matrix"""
        return self.all_cities.index(city)
    
    def generate_partitions(self, n_loops: int):
        """Generate all partitions of cities into n_loops non-empty sets"""
        items = sorted(self.cities)
        
        if n_loops < 1 or n_loops > len(items):
            return
        
        def helper(rest, k):
            if k == 1:
                yield [set(rest)]
                return
            
            anchor = rest[0]
            tail = rest[1:]
            max_r = len(tail) - (k - 1)
            
            if max_r < 0:
                return
            
            for r in range(0, max_r + 1):
                for extra in combinations(tail, r):
                    block = {anchor, *extra}
                    extra_set = set(extra)
                    remaining = [x for x in tail if x not in extra_set]
                    
                    for rest_blocks in helper(remaining, k - 1):
                        yield [block] + rest_blocks
        
        yield from helper(items, n_loops)
    
    def find_optimal_sequence(self, cities: Set[str]) -> Tuple[List[str], float]:
        """Find optimal TSP tour for given cities"""
        if not cities:
            return [], 0.0
        
        key = frozenset(cities)
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
            route, dist = self._tsp_held_karp(city_list)
            self._hk_cache[key] = (list(route), dist)
            return route, dist
        else:
            route = self._nearest_insertion_route(city_list)
            route = self._two_opt_single_pass(route)
            dist = float(self.calculate_route_distance(route))
            self._heur_cache[key] = (list(route), dist)
            return route, dist
    
    def _idx(self, node: str) -> int:
        return 0 if node == 'WHS' else self.get_city_index(node)
    
    def _d(self, a: str, b: str) -> float:
        return float(self.distances[self._idx(a)][self._idx(b)])
    
    def _tsp_held_karp(self, cities: List[str]) -> Tuple[List[str], float]:
        """Exact TSP using Held-Karp algorithm"""
        idxs = [0] + [self.get_city_index(c) for c in cities]
        m = len(cities)
        D = [[self.distances[i][j] for j in idxs] for i in idxs]
        
        INF = float('inf')
        size = 1 << m
        dp = [[INF]*(m+1) for _ in range(size)]
        par = [[(-1, -1)]*(m+1) for _ in range(size)]
        
        for j in range(1, m+1):
            dp[1 << (j-1)][j] = D[0][j]
        
        for mask in range(size):
            for j in range(1, m+1):
                if not (mask & (1 << (j-1))):
                    continue
                prev_mask = mask ^ (1 << (j-1))
                if prev_mask == 0:
                    continue
                best = dp[mask][j]
                best_k = -1
                kbits = prev_mask
                while kbits:
                    ls = kbits & -kbits
                    k = (ls.bit_length())
                    kbits -= ls
                    cand = dp[prev_mask][k] + D[k][j]
                    if cand < best:
                        best = cand
                        best_k = k
                if best < dp[mask][j]:
                    dp[mask][j] = best
                    par[mask][j] = (prev_mask, best_k)
        
        full = size - 1
        best_end, best_total = -1, INF
        for j in range(1, m+1):
            total = dp[full][j] + D[j][0]
            if total < best_total:
                best_total, best_end = total, j
        
        route_idx = []
        mask, j = full, best_end
        while j != -1 and mask:
            route_idx.append(j)
            mask, j = par[mask][j]
        route_idx.reverse()
        route = [cities[k-1] for k in route_idx]
        return route, float(best_total)
    
    def _nearest_insertion_route(self, cities: List[str]) -> List[str]:
        """Nearest insertion heuristic"""
        unvisited = set(cities)
        
        c1 = min(unvisited, key=lambda c: (self._d('WHS', c), c))
        unvisited.remove(c1)
        if not unvisited:
            return [c1]
        
        c2 = min(unvisited, key=lambda c: (self._d(c1, c), c))
        unvisited.remove(c2)
        
        cycle_len_12 = self._d('WHS', c1) + self._d(c1, c2) + self._d(c2, 'WHS')
        cycle_len_21 = self._d('WHS', c2) + self._d(c2, c1) + self._d(c1, 'WHS')
        route = [c1, c2] if cycle_len_12 <= cycle_len_21 else [c2, c1]
        
        def best_insertion(route, u):
            best_delta = float('inf')
            best_pos = 0
            
            candidates = [('WHS', route[0], 0)]
            candidates += [(route[i], route[i+1], i+1) for i in range(len(route)-1)]
            candidates += [(route[-1], 'WHS', len(route))]
            
            for a, b, pos in candidates:
                delta = self._d(a, u) + self._d(u, b) - self._d(a, b)
                if (delta < best_delta) or (abs(delta - best_delta) < 1e-12 and pos < best_pos):
                    best_delta, best_pos = delta, pos
            return best_pos, best_delta
        
        while unvisited:
            best_city = None
            best_pos = 0
            best_delta = float('inf')
            for u in sorted(unvisited):
                pos, delta = best_insertion(route, u)
                if (delta < best_delta or
                    (abs(delta - best_delta) < 1e-12 and 
                     (u < (best_city or u) or (u == best_city and pos < best_pos)))):
                    best_city, best_pos, best_delta = u, pos, delta
            route.insert(best_pos, best_city)
            unvisited.remove(best_city)
        
        return route
    
    def _two_opt_single_pass(self, route: List[str]) -> List[str]:
        """Single pass 2-opt improvement"""
        n = len(route)
        if n < 3:
            return route
        
        ext = ['WHS'] + route[:] + ['WHS']
        best_ext = ext
        best_improved = False
        
        for i in range(0, n):
            for j in range(i+1, n+1):
                a, b = ext[i], ext[i+1]
                c, d = ext[j], ext[j+1]
                old = self._d(a, b) + self._d(c, d)
                new = self._d(a, c) + self._d(b, d)
                if new + 1e-12 < old:
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
        """Calculate total distance for a route"""
        total = 0
        current = 0
        
        for city in route:
            next_idx = self.get_city_index(city)
            total += self.distances[current][next_idx]
            current = next_idx
        
        total += self.distances[current][0]
        return total
    
    def optimize_day(self, day: int, record_size: int = 10, max_loops: int = 6) -> List[DayStrategy]:
        """Optimize for a single day and return top strategies"""
        strategies = []
        
        for n_loops in range(1, max_loops + 1):
            partitions = self.generate_partitions(n_loops)
            
            for partition in partitions:
                loops = []
                valid = True
                
                for subset in partition:
                    sequence, distance = self.find_optimal_sequence(subset)
                    
                    # Calculate loop properties
                    n_cities = len(sequence)
                    time_days = 0.00234 * distance + 0.053 * n_cities
                    duration_days = math.ceil(time_days)
                    
                    # Check demand constraint
                    loop_demand = sum(self.demands[day][self.cities.index(city)] 
                                    for city in sequence)
                    if loop_demand > self.truck_capacity:
                        valid = False
                        break
                    
                    loop = Loop(
                        route=sequence,
                        distance=distance,
                        duration_days=duration_days,
                        time_days=time_days,
                        cities=set(sequence)
                    )
                    loops.append(loop)
                
                if not valid:
                    continue
                
                # Simple scoring for single day (just for ranking)
                total_duration = sum(l.duration_days for l in loops)
                if total_duration > 10:  # Basic truck constraint
                    continue
                
                # Create strategy
                strategy = DayStrategy(
                    loops=loops,
                    score=0,  # Will be properly scored in multi-day
                    indexes={},
                    day_type=day % 5
                )
                strategies.append(strategy)
        
        # Sort by simple heuristic (minimize total duration and distance)
        strategies.sort(key=lambda s: (
            sum(l.duration_days for l in s.loops),
            sum(l.distance for l in s.loops)
        ))
        
        return strategies[:record_size]


class MultiDayOptimizer:
    """Multi-day optimization with truck resource management"""
    
    def __init__(self):
        # City and distance setup
        self.cities = ['BNG', 'JAK', 'ORL', 'TPA', 'ATL', 'MAC', 'BTM', 
                      'CHR', 'RAL', 'COL', 'MPH', 'NSH', 'ALX', 'RCH']
        self.all_cities = ['WHS'] + self.cities
        
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
        self.num_trucks = 10
        self.horizon_days = 10
        self.record_size = 5  # Top K strategies per weekday
        
        # Corrected weights
        self.w1 = 4431    # NT (max trucks)
        self.w2 = 0.345   # ML (total mileage)
        self.w3 = 690     # TDD (total driver days)
        self.w4 = 0   # ACW (average customers waiting)
        self.w5 = 0    # MCW (max customers waiting) - fixed at 14
        
        # Single day optimizer
        self.single_optimizer = SingleDayOptimizer(
            self.distances, self.demands, self.cities, self.all_cities
        )
    
    def get_weekday_strategies(self, verbose: bool = True) -> Dict[int, List[DayStrategy]]:
        """Get top strategies for each weekday"""
        weekday_strategies = {}
        
        if verbose:
            print("="*60)
            print("Phase 1: Single-Day Optimization")
            print("="*60)
        
        for day in range(5):  # Monday to Friday
            if verbose:
                weekday_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][day]
                print(f"\nOptimizing for {weekday_name} (Day {day})...")
            
            strategies = self.single_optimizer.optimize_day(
                day=day, 
                record_size=self.record_size,
                max_loops=6
            )
            
            weekday_strategies[day] = strategies
            
            if verbose:
                print(f"  Found {len(strategies)} top strategies")
                if strategies:
                    print(f"  Best: {len(strategies[0].loops)} loops, "
                          f"total distance: {sum(l.distance for l in strategies[0].loops):.1f} miles")
        
        return weekday_strategies
    
    def simulate_composition(self, composition: List[DayStrategy]) -> Optional[Dict]:
        """
        Simulate a 10-day schedule with the given weekday composition.
        Returns metrics if feasible, None if infeasible.
        """
        # Initialize trucks
        trucks = [TruckState(id=i, busy_until_day=-1) for i in range(self.num_trucks)]
        
        # Tracking metrics
        daily_active_trucks = []
        total_mileage = 0
        total_driver_days = 0
        daily_waiting = []
        max_trucks_used = 0
        
        # Simulate each day
        for day in range(self.horizon_days):
            weekday = day % 5
            strategy = composition[weekday]
            
            # Free up trucks that finished
            available_trucks = [t for t in trucks if t.is_available(day)]
            busy_trucks = [t for t in trucks if not t.is_available(day)]
            
            # Check if we can launch all loops
            if len(strategy.loops) > len(available_trucks):
                return None  # Infeasible
            
            # Launch loops
            for i, loop in enumerate(strategy.loops):
                truck = available_trucks[i]
                truck.busy_until_day = day + loop.duration_days - 1
                truck.current_loop = loop
                total_mileage += loop.distance
            
            # Track active trucks
            active_count = len(busy_trucks) + len(strategy.loops)
            daily_active_trucks.append(active_count)
            max_trucks_used = max(max_trucks_used, active_count)
            
            # Calculate waiting (simplified - all cities in today's loops start waiting)
            day_waiting = sum(len(loop.cities) for loop in strategy.loops)
            daily_waiting.append(day_waiting)
        
        # Calculate final metrics
        total_driver_days = sum(daily_active_trucks)
        avg_waiting = np.mean(daily_waiting) if daily_waiting else 0
        max_waiting = 14  # Fixed as specified
        
        return {
            'max_trucks': max_trucks_used,
            'total_mileage': total_mileage,
            'total_driver_days': total_driver_days,
            'avg_waiting': avg_waiting,
            'max_waiting': max_waiting,
            'daily_active': daily_active_trucks,
            'feasible': True
        }
    
    def calculate_score(self, metrics: Dict) -> float:
        """Calculate the multi-day score"""
        return (self.w1 * metrics['max_trucks'] +
                self.w2 * metrics['total_mileage'] +
                self.w3 * metrics['total_driver_days'] +
                self.w4 * metrics['avg_waiting'] +
                self.w5 * metrics['max_waiting'])
    
    def optimize(self, verbose: bool = True) -> Tuple[List[DayStrategy], Dict, float]:
        """Main multi-day optimization"""
        # Step 1: Get best strategies for each weekday
        weekday_strategies = self.get_weekday_strategies(verbose)
        
        if verbose:
            print("\n" + "="*60)
            print("Phase 2: Multi-Day Composition Search")
            print("="*60)
            total_compositions = self.record_size ** 5
            print(f"\nTotal compositions to evaluate: {total_compositions:,}")
        
        # Step 2: Generate all compositions
        best_composition = None
        best_metrics = None
        best_score = float('inf')
        
        # Get strategy lists for each weekday
        monday_strats = weekday_strategies[0]
        tuesday_strats = weekday_strategies[1]
        wednesday_strats = weekday_strategies[2]
        thursday_strats = weekday_strategies[3]
        friday_strats = weekday_strategies[4]
        
        evaluated = 0
        feasible_count = 0
        
        # Try all combinations
        for comp in product(monday_strats, tuesday_strats, wednesday_strats, 
                           thursday_strats, friday_strats):
            evaluated += 1
            
            if verbose and evaluated % 100 == 0:
                print(f"  Evaluated: {evaluated}/{total_compositions} "
                      f"({100*evaluated/total_compositions:.1f}%), "
                      f"Feasible: {feasible_count}")
            
            # Simulate this composition
            metrics = self.simulate_composition(list(comp))
            
            if metrics is None:
                continue  # Infeasible
            
            feasible_count += 1
            score = self.calculate_score(metrics)
            
            if score < best_score:
                best_score = score
                best_composition = list(comp)
                best_metrics = metrics
        
        if verbose:
            print(f"\nEvaluation complete!")
            print(f"  Total evaluated: {evaluated}")
            print(f"  Feasible solutions: {feasible_count}")
            print(f"  Best score: {best_score:.2f}")
        
        return best_composition, best_metrics, best_score
    
    def print_solution(self, composition: List[DayStrategy], 
                      metrics: Dict, score: float):
        """Print the solution in a readable format"""
        print("\n" + "="*60)
        print("OPTIMAL MULTI-DAY SOLUTION")
        print("="*60)
        
        print(f"\nOverall Score: {score:.2f}")
        
        print("\nWeekday Strategy Selection:")
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for i, strategy in enumerate(composition):
            print(f"\n{weekdays[i]}:")
            for j, loop in enumerate(strategy.loops, 1):
                route_str = " → ".join(['WHS'] + loop.route + ['WHS'])
                print(f"  Loop {j}: {route_str}")
                print(f"          Duration: {loop.duration_days} days, "
                      f"Distance: {loop.distance:.1f} miles")
        
        print("\nPerformance Metrics:")
        print(f"  • Maximum Trucks Used:        {metrics['max_trucks']}")
        print(f"  • Total Mileage:             {metrics['total_mileage']:.1f} miles")
        print(f"  • Total Driver Days:         {metrics['total_driver_days']}")
        print(f"  • Average Customers Waiting: {metrics['avg_waiting']:.2f}")
        print(f"  • Maximum Customers Waiting: {metrics['max_waiting']}")
        
        print("\nDaily Truck Usage:")
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        for i, active in enumerate(metrics['daily_active']):
            print(f"  Day {i+1:2} ({days[i]}): {active} trucks active")
        
        print("\nScore Breakdown:")
        print(f"  • Trucks:    {self.w1} × {metrics['max_trucks']} = "
              f"{self.w1 * metrics['max_trucks']:.2f}")
        print(f"  • Mileage:   {self.w2} × {metrics['total_mileage']:.1f} = "
              f"{self.w2 * metrics['total_mileage']:.2f}")
        print(f"  • Driver Days: {self.w3} × {metrics['total_driver_days']} = "
              f"{self.w3 * metrics['total_driver_days']:.2f}")
        print(f"  • Avg Wait:  {self.w4} × {metrics['avg_waiting']:.2f} = "
              f"{self.w4 * metrics['avg_waiting']:.2f}")
        print(f"  • Max Wait:  {self.w5} × {metrics['max_waiting']} = "
              f"{self.w5 * metrics['max_waiting']:.2f}")


def main():
    """Main execution"""
    print("="*60)
    print("MULTI-DAY TRANSPORTATION OPTIMIZATION SYSTEM")
    print("="*60)
    print("\nInitializing optimizer...")
    
    optimizer = MultiDayOptimizer()
    
    print("Starting optimization process...")
    start_time = time.time()
    
    # Run optimization
    best_composition, best_metrics, best_score = optimizer.optimize(verbose=True)
    
    end_time = time.time()
    
    # Print results
    if best_composition:
        optimizer.print_solution(best_composition, best_metrics, best_score)
    else:
        print("\nNo feasible solution found!")
    
    print(f"\n{'='*60}")
    print(f"Total optimization time: {end_time - start_time:.2f} seconds")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
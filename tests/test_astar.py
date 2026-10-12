"""Exercise the public search function on Python 3."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astar import G, astar


class AstarTests(unittest.TestCase):
    def search(self, graph, start, target, heuristic=None, **kwargs):
        return astar(
            start,
            lambda pos: graph.get(pos, {}),
            lambda pos: pos == target,
            0,
            lambda a, b: graph[a][b],
            heuristic or (lambda pos: 0),
            **kwargs
        )

    def test_weighted_shortest_path_excludes_start(self):
        graph = {"start": {"goal": 10, "via": 1}, "via": {"goal": 2}}
        self.assertEqual(self.search(graph, "start", "goal"), ["via", "goal"])

    def test_start_is_goal(self):
        self.assertEqual(self.search({}, "start", "start"), [])

    def test_unreachable_goal_returns_closest_known_path(self):
        graph = {"start": {"near": 1}, "near": {"end": 1}}
        estimates = {"start": 3, "near": 1, "end": 2}
        self.assertEqual(
            self.search(graph, "start", "goal", estimates.__getitem__), ["near"]
        )

    def test_limit_stops_discovery(self):
        graph = {"start": {"near": 1}, "near": {"goal": 1}}
        estimates = {"start": 2, "near": 1, "goal": 0}
        self.assertEqual(
            self.search(graph, "start", "goal", estimates.__getitem__, limit=2),
            ["near"],
        )

    def test_equal_priorities_support_unorderable_positions(self):
        start, left, right, target = [object() for _ in range(4)]
        graph = {start: {left: 1, right: 1}, left: {target: 1}, right: {target: 1}}
        self.assertEqual(self.search(graph, start, target), [left, target])

    def test_open_node_is_replaced_when_a_cheaper_path_is_found(self):
        graph = {"start": {"via": 1, "other": 8}, "via": {"other": 1}}
        nodes = {}
        self.search(graph, "start", "missing", debug=nodes.update)
        self.assertEqual(nodes["other"][G], 2)

    def test_closed_node_is_reopened_with_an_inconsistent_heuristic(self):
        graph = {
            "start": {"a": 3, "b": 1},
            "a": {"goal": 3},
            "b": {"a": 1},
        }
        estimates = {"start": 5, "a": 0, "b": 4, "goal": 0}
        self.assertEqual(
            self.search(graph, "start", "goal", estimates.__getitem__),
            ["b", "a", "goal"],
        )

    def test_debug_receives_start_cost_and_discovered_nodes(self):
        nodes = {}
        path = astar(
            "start", lambda pos: ["goal"] if pos == "start" else [],
            lambda pos: pos == "goal", 5, lambda a, b: 2,
            lambda pos: 0, debug=nodes.update,
        )
        self.assertEqual(path, ["goal"])
        self.assertEqual(nodes["start"][G], 5)
        self.assertEqual(nodes["goal"][G], 7)


if __name__ == "__main__":
    unittest.main()

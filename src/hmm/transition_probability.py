"""Transition probabilities between consecutive candidate road arcs."""

from functools import lru_cache

import networkx as nx
import numpy as np


class TransitionModel:
    def __init__(self, trip):
        self.trip = trip

        # Directed road network. Arc lengths are measured in metres.
        self.graph = nx.DiGraph()

        for arc_id, (start, end) in enumerate(trip.arcs):
            a = trip.nodes_xy[start]
            b = trip.nodes_xy[end]
            length = float(np.linalg.norm(b - a))

            if not self.graph.has_edge(start, end):
                self.graph.add_edge(
                    start,
                    end,
                    weight=length
                )

        self.arc_lengths = np.linalg.norm(
            trip.nodes_xy[trip.arcs[:, 1]]
            - trip.nodes_xy[trip.arcs[:, 0]],
            axis=1
        )

    @lru_cache(maxsize=10000)
    def shortest_path(self, source, destination):
        """Shortest directed road-network distance between two nodes."""

        try:
            return nx.shortest_path_length(
                self.graph,
                source=int(source),
                target=int(destination),
                weight="weight"
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def transition_score(
        self,
        previous_arc,
        current_arc,
        previous_fraction,
        current_fraction,
        gps_distance,
        time_gap
    ):
        """Return an unnormalised transition score."""

        previous_arc = int(previous_arc)
        current_arc = int(current_arc)

        previous_start, previous_end = self.trip.arcs[previous_arc]
        current_start, current_end = self.trip.arcs[current_arc]

        previous_length = self.arc_lengths[previous_arc]
        current_length = self.arc_lengths[current_arc]

        previous_position = previous_fraction * previous_length
        current_position = current_fraction * current_length

        # Case 1: Both observations map to the same road segment.
        if previous_arc == current_arc:
            network_distance = max(
                0.0,
                current_position - previous_position
            )

        else:
            # Distance remaining on previous arc
            remaining = previous_length - previous_position

            # Distance between the two arcs through the road network
            middle_distance = self.shortest_path(
                int(previous_end),
                int(current_start)
            )

            if middle_distance is None:
                # Disconnected candidates receive a strong penalty.
                return 1e-8

            network_distance = (
                remaining
                + middle_distance
                + current_position
            )

        # Allow greater movement during larger GPS time gaps.
        time_gap = max(float(time_gap), 1.0)

        tolerance = max(
            20.0,
            10.0 + 0.5 * max(gps_distance, 0.0)
        )

        tolerance *= min(np.sqrt(time_gap), 5.0)

        # Penalise disagreement between road distance and GPS displacement.
        difference = abs(network_distance - gps_distance)

        score = np.exp(-difference / tolerance)

        return max(float(score), 1e-8)
from itertools import combinations

import matplotlib
import polars as pl

matplotlib.use("Agg")

from wisco_slap.util.plot import (  # noqa: E402
    DMD1_DENDRITE_COLOR,
    DMD2_DENDRITE_COLOR,
    SOMA_TREE_COLOR,
    plot_event_dendrite_tree,
)


def test_plot_event_dendrite_tree_counts_unique_synapses_and_colors_dmds() -> None:
    events = pl.DataFrame({
        "dmd": [1, 1, 1, 2, 2, 2, 2],
        "syn_id": [10, 10, 11, 20, 21, 21, 30],
        "soma-ID": ["soma1", "soma1", "soma1", "soma1", "soma1", "soma1", "soma2"],
        "soma-depth": [235.0, 235.0, 235.0, 235.0, 235.0, 235.0, 180.0],
        "dend-ID": ["A-1", "A-1", "A-1", "B-1", "B-1", "B-1", "C-1"],
    })

    dmd_info = {
        "mouse": {
            "exp_1": {
                "loc_A": {
                    "acq_1": {
                        "dmd-1": {"depth": 235, "somas": ["soma1"]},
                        "dmd-2": {"depth": 180, "somas": ["soma2"]},
                    }
                }
            }
        }
    }

    fig, ax = plot_event_dendrite_tree(
        events,
        subject="mouse",
        exp="exp_1",
        loc="loc_A",
        acq="acq_1",
        dmd_info=dmd_info,
        title=None,
    )

    texts = {text.get_text(): text for text in ax.texts}
    assert "soma1\n(DMD1, depth=235um)" in texts
    assert "soma2\n(DMD2, depth=180um)" in texts
    assert "soma1\n(DMD1/DMD2, depth=235um)" not in texts
    assert "A-1\n(DMD1, 2 syns)" in texts
    assert "B-1\n(DMD2, 2 syns)" in texts
    assert "C-1\n(DMD2, 1 syn)" in texts

    assert texts["soma1\n(DMD1, depth=235um)"].get_color() == SOMA_TREE_COLOR
    assert texts["A-1\n(DMD1, 2 syns)"].get_color() == DMD1_DENDRITE_COLOR
    assert texts["B-1\n(DMD2, 2 syns)"].get_color() == DMD2_DENDRITE_COLOR

    fig.clear()


def test_plot_event_dendrite_tree_rejects_duplicate_soma_dmd_metadata() -> None:
    events = pl.DataFrame({
        "dmd": [1],
        "syn_id": [10],
        "soma-ID": ["soma1"],
        "soma-depth": [235.0],
        "dend-ID": ["A-1"],
    })
    dmd_info = {
        "mouse": {
            "exp_1": {
                "loc_A": {
                    "acq_1": {
                        "dmd-1": {"depth": 235, "somas": ["soma1"]},
                        "dmd-2": {"depth": 180, "somas": ["soma1"]},
                    }
                }
            }
        }
    }

    try:
        plot_event_dendrite_tree(
            events,
            subject="mouse",
            exp="exp_1",
            loc="loc_A",
            acq="acq_1",
            dmd_info=dmd_info,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected duplicate soma metadata ValueError")

    assert "listed under both DMD1 and DMD2" in message


def test_plot_event_dendrite_tree_avoids_soma_label_overlap() -> None:
    events = pl.DataFrame({
        "dmd": [1, 2, 2, 1, 1],
        "syn_id": [10, 20, 30, 40, 50],
        "soma-ID": [
            "soma1",
            "soma2",
            "somaHUKB",
            "unidentifiable_soma",
            "very_long_manual_soma_label",
        ],
        "soma-depth": [145.0, 90.0, 170.0, None, 210.0],
        "dend-ID": ["A-1", "B-1", "K-1", "D-1", "E-1"],
    })
    dmd_info = {
        "mouse": {
            "exp_1": {
                "loc_A": {
                    "acq_1": {
                        "dmd-1": {
                            "depth": 145,
                            "somas": [
                                "soma1",
                                "unidentifiable_soma",
                                "very_long_manual_soma_label",
                            ],
                        },
                        "dmd-2": {"depth": 90, "somas": ["soma2", "somaHUKB"]},
                    }
                }
            }
        }
    }

    fig, ax = plot_event_dendrite_tree(
        events,
        subject="mouse",
        exp="exp_1",
        loc="loc_A",
        acq="acq_1",
        dmd_info=dmd_info,
        title=None,
        max_soma_id_line_chars=12,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    soma_boxes = [
        text.get_window_extent(renderer).expanded(1.02, 1.02)
        for text in ax.texts
        if text.get_color() == SOMA_TREE_COLOR
    ]

    for left, right in combinations(soma_boxes, 2):
        assert not left.overlaps(right)

    fig.clear()


def test_plot_event_dendrite_tree_reports_missing_columns() -> None:
    events = pl.DataFrame({"syn_id": [1]})

    try:
        plot_event_dendrite_tree(events)
    except KeyError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected missing column KeyError")

    assert "soma-ID" in message
    assert "dend-ID" in message

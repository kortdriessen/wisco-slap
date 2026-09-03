"""Experimental subgroup sets — named, human-reviewed cohorts of data.

See ``subgroup_info.md`` in this directory for the full workflow.
"""

from ._get_subgroups import list_subgroup_sets, load_subgroup_set

__all__ = ["list_subgroup_sets", "load_subgroup_set"]

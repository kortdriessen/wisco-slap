"""Experimental subgroup sets — named cohorts of data defined by explicit criteria.

See ``subgroup_info.md`` in this directory for the full workflow.
"""

from ._get_subgroups import list_subgroup_sets, load_subgroup_set, members_to_df

__all__ = ["list_subgroup_sets", "load_subgroup_set", "members_to_df"]

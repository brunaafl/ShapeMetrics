import os
from collections import defaultdict
import numpy as np
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Specify path to data.
DATAPATH = "data/"

# Add path to save data summaries.
if not os.path.exists(
        os.path.join(DATAPATH, "extracted_data")
    ):
    os.mkdir(os.path.join(DATAPATH, "extracted_data"))

# Create object storing references to online data.
data_cache = EcephysProjectCache.from_warehouse(
    manifest=os.path.join(DATAPATH, "manifest.json")
)

# Load reference space.
from allensdk.core.reference_space_cache import ReferenceSpaceCache
REF_SPACE_RESOLUTION = 10
REF_SPACE_KEY = "annotation/ccf_2017"
rspc = ReferenceSpaceCache(
    REF_SPACE_RESOLUTION,
    REF_SPACE_KEY,
    manifest=os.path.join(DATAPATH, "rsp_manifest.json"),
)
rsp = rspc.get_reference_space()
tree = rspc.get_structure_tree(structure_graph_id=1)

# Collect session metadata. Filter to brain observatory 1.1 stimulus set.
sess_table = data_cache.get_session_table()
sess_table = sess_table[(sess_table["session_type"] == "brain_observatory_1.1")]

print("total of %i sessions" % len(sess_table.index.values))

# Extract responses for natural movies and static gratings.
for stim_name, stim_duration in [
        ("natural_movie_three", 0.033355),
        ("static_gratings", 0.249),
    ]:

    # Dictionary containing the session and unit ids we extract.
    sess_plus_cell_ids = defaultdict(list)
    responses = defaultdict(list)

    # Iterate over recording session, pooling data from same brain regions.
    for sess_id in sess_table.index.values:

        # Print progress.
        print(f"Processing session {sess_id}...")

        # Load session.
        session = data_cache.get_session_data(sess_id)
        continue
        # Collect units.
        units = session.units.dropna(subset=[
            "anterior_posterior_ccf_coordinate",
            "dorsal_ventral_ccf_coordinate",
            "left_right_ccf_coordinate"
        ])
        n_units = units.index.values.size
        if n_units == 0:
            continue

        # Find cell locations.
        _ap = (units["anterior_posterior_ccf_coordinate"] / REF_SPACE_RESOLUTION).astype("int")
        _dv = (units["dorsal_ventral_ccf_coordinate"] / REF_SPACE_RESOLUTION).astype("int")
        _lr = (units["left_right_ccf_coordinate"] / REF_SPACE_RESOLUTION).astype("int")
        cell_locs = tree.get_structures_by_id(
            [rsp.annotation[ap, dv, lr] for ap, dv, lr in zip(_ap, _dv, _lr)]
        )
        brain_region_acronyms = [s["acronym"] for s in cell_locs]

        # Store unit ids alongside session id in each brain region.
        for acronym, u in zip(brain_region_acronyms, units.index.values):
            sess_plus_cell_ids[acronym].append((sess_id, u))

        # Extract stimulus presentation ids
        m = (session.stimulus_presentations["stimulus_name"] == stim_name)
        stim_idx = session.stimulus_presentations[m].index.values

        # Extract activity from all units.
        Xraw = session.presentationwise_spike_counts(
            bin_edges=[0., stim_duration],
            stimulus_presentation_ids=stim_idx,
            unit_ids=units.index.values
        ).values[:, 0, :]

        # Determine unique stimulus conditons.
        stim_conditions= np.unique(
            session.stimulus_presentations[m].stimulus_condition_id
        )
        n_conditions = len(stim_conditions)

        # Average responses over trials within each condition.
        X = np.empty((n_conditions, n_units))
        for ci, c in enumerate(stim_conditions):
            X[ci] = np.nanmean(
                Xraw[session.stimulus_presentations[m].stimulus_condition_id == c],
                axis=0
            )

        # Group neural responses by brain structure.
        for n, acronym in enumerate(brain_region_acronyms):
            responses[acronym].append(X[:, n])

    # Save session and cell ids for each unit.
    np.savez(
        os.path.join(
            DATAPATH, "extracted_data", f"{stim_name}_sess_plus_cell_ids.npz"
        ), **sess_plus_cell_ids
    )

    #Save stimulus condition ids.
    np.save(
        os.path.join(
            DATAPATH, "extracted_data", f"{stim_name}_stim_conditions.npy"
        ), stim_conditions
    )

    # Save cell responses.
    np.savez(
        os.path.join(
            DATAPATH, "extracted_data", f"{stim_name}_responses.npz"
        ), **responses
    )


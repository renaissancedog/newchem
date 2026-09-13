import pandas as pd
import logging

try:
    from mp_api.client import MPRester
except ImportError:  # mp-api is only needed for the retrieval functions below, and is
    MPRester = None  # deliberately not required to run relaxations on a compute node.

# Configure logger for this module
logger = logging.getLogger(__name__)


def _require_mprester():
    if MPRester is None:
        raise ImportError(
            "mp-api is required for Materials Project retrieval. Install it with "
            "`pip install mp-api`, or use the calculation/intercalation modules, which "
            "do not need it."
        )


def get_electrode_pairs(working_ion: str, api_key: str | None, fields: list) -> pd.DataFrame:
    """
    Retrieves electrode pairs data (charge and discharge IDs) from the Materials Project
    for a given working ion. Saves the data to an HDF5 file.

    Args:
        working_ion (str): The working ion (e.g., "Li", "Na").
        api_key (str | None): Materials Project API key.
        fields (list): List of fields to retrieve from the insertion_electrodes search.

    Returns:
        pd.DataFrame: DataFrame containing 'charge_id' and 'discharge_id' for electrode pairs.
                      Returns an empty DataFrame if an error occurs or no data is found.
    """

    _require_mprester()
    found_pairs_data = []
    df_found_pairs = pd.DataFrame()
    try:
        with MPRester(api_key=api_key) as mpr:
            logger.debug("Available insertion electrode fields: "
                         f"{mpr.materials.insertion_electrodes.available_fields}")
            logger.info(f"Querying Materials Project for {working_ion}-ion insertion electrodes...")

            electrode_docs = mpr.materials.insertion_electrodes.search(working_ion=working_ion, fields=fields)
            logger.info(f"Retrieved {len(electrode_docs)} initial insertion electrode entries from MP.")

            for doc in electrode_docs:
                if (doc.id_charge and doc.id_discharge) and doc.working_ion:  # Ensure both IDs are present
                    found_pairs_data.append({
                        "charge_id": doc.id_charge,
                        "discharge_id": doc.id_discharge,
                        "working_ion": doc.working_ion
                    })
                else:
                    logger.debug(f"Skipping entry due to missing charge/discharge ID or working_ion: {doc}")

        if found_pairs_data:
            df_found_pairs = pd.DataFrame(found_pairs_data)
            logger.info(f"Found {len(df_found_pairs)} unique electrode pairs.")
        else:
            logger.warning(f"No electrode pairs data retrieved or processed for {working_ion}-ion.")

    except Exception as e:
        logger.error(f"An error occurred during electrode pair retrieval: {e}")

    return df_found_pairs


def get_structures_from_electrode_pair_ids(df_pairs: pd.DataFrame, api_key: str | None, fields: list) -> None:
    """
    Retrieves Pymatgen Structure objects and related information (formula, energy per atom)
    for charge and discharge IDs present in the input DataFrame.
    Modifies the input DataFrame `df_pairs` in place by adding new columns:
    'charge_structure', 'charge_formula', 'charge_energy_per_atom',
    'discharge_structure', 'discharge_formula', 'discharge_energy_per_atom'.

    Args:
        df_pairs (pd.DataFrame): DataFrame containing 'charge_id' and 'discharge_id' columns.
                                 This DataFrame will be modified.
        api_key (str | None): Materials Project API key.
        fields (list): List of fields to retrieve from the materials summary search.
    """
    _require_mprester()
    if df_pairs.empty:
        logger.warning("Input DataFrame `df_pairs` is empty. Skipping structure retrieval.")
        return

    # Get unique material IDs to minimize API calls
    charge_ids = df_pairs['charge_id'].dropna().unique().tolist()
    discharge_ids = df_pairs['discharge_id'].dropna().unique().tolist()
    all_unique_ids = list(set(charge_ids + discharge_ids))

    if not all_unique_ids:
        logger.info("No valid material IDs found in DataFrame. Skipping structure retrieval.")
        return

    logger.info(f"Retrieving structures for {len(all_unique_ids)} unique material IDs...")
    BATCH_SIZE = 5000
    material_docs_map = {}
    try:
        with MPRester(api_key=api_key) as mpr:
            for i in range(0, len(all_unique_ids), BATCH_SIZE):
                batch_ids = all_unique_ids[i:i + BATCH_SIZE]
                retrieved_docs = mpr.materials.summary.search(material_ids=batch_ids, fields=fields)

                for doc in retrieved_docs:
                    material_docs_map[doc.material_id] = doc
            logger.info(f"Retrieved data for {len(material_docs_map)} out of {len(all_unique_ids)} "
                        "requested material IDs.")

    except Exception as e:
        logger.error(f"An error occurred during structure retrieval from MP: {e}")
        return

    # Helper function to map data
    def map_structure_data(material_id, docs_map):
        doc = docs_map.get(material_id)
        return_list = []
        for field in fields:
            return_list.append(getattr(doc, field, None) if doc else None)
        return_list.append(doc.structure.composition.formula if doc else None)  # Add formula of the structure
        return return_list

    # Apply mapping to DataFrame columns
    charge_data = df_pairs['charge_id'].apply(
        lambda mid: pd.Series(map_structure_data(mid, material_docs_map), index=fields+['formula']))
    df_pairs[['charge_'+field for field in fields]+['charge_formula']] = charge_data

    discharge_data = df_pairs['discharge_id'].apply(
        lambda mid: pd.Series(map_structure_data(mid, material_docs_map), index=fields+['formula']))
    df_pairs[['discharge_'+field for field in fields]+['discharge_formula']] = discharge_data

    logger.info(f"Successfully added {df_pairs['charge_structure'].notna().sum()} charge structures.")
    logger.info(f"Successfully added {df_pairs['discharge_structure'].notna().sum()} discharge structures.")


def get_materials_summary(api_key: str | None, fields: list, **search_kwargs) -> pd.DataFrame:
    """
    Retrieves materials from the Materials Project summary endpoint as a DataFrame.

    This is a thin, general wrapper around `mpr.materials.summary.search`: any filter the
    endpoint accepts (e.g. `theoretical=False`, `energy_above_hull=(0, 0.1)`) can be passed
    through `search_kwargs`, so the selection criteria live in the calling script rather
    than in the library.

    Structures are returned as JSON strings (not Structure objects) so that the resulting
    DataFrame round-trips through HDF5 without pickling, and stays readable across
    pymatgen and numpy versions.

    Args:
        api_key (str | None): Materials Project API key.
        fields (list): Fields to request. "material_id" and "structure" are recommended.
        **search_kwargs: Filters forwarded to `mpr.materials.summary.search`.

    Returns:
        pd.DataFrame: One row per material, with a column per requested field. The
                      'structure' column, if requested, is renamed to 'mp_structure' and
                      accompanied by 'num_sites'. Empty DataFrame on error.
    """
    _require_mprester()
    try:
        with MPRester(api_key=api_key) as mpr:
            logger.info(f"Querying Materials Project summary endpoint with {search_kwargs}...")
            docs = mpr.materials.summary.search(fields=fields, **search_kwargs)
            logger.info(f"Retrieved {len(docs)} materials from MP.")
    except Exception as e:
        logger.error(f"An error occurred during materials retrieval: {e}")
        return pd.DataFrame()

    rows = []
    for doc in docs:
        try:
            row = {}
            for field in fields:
                value = getattr(doc, field, None)
                if field == "structure" and value is not None:
                    row["mp_structure"] = value.to(fmt="json")
                    row["num_sites"] = len(value)
                elif field == "material_id":
                    row["material_id"] = str(value) if value is not None else None
                else:
                    row[field] = value
            rows.append(row)
        except Exception as e:
            logger.warning(f"Error processing {getattr(doc, 'material_id', 'unknown')}: {e}")
            continue

    df = pd.DataFrame(rows)
    logger.info(f"Built DataFrame with {len(df)} materials and columns {list(df.columns)}.")
    return df

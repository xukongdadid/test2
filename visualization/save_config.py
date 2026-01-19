DOF_KEYS = [
    "dof_Surge", "dof_Sway", "dof_Heave", "dof_Roll", "dof_Pitch", "dof_Yaw",
    "dof_TwrFA1", "dof_TwrSS1", "dof_TwrFA2", "dof_TwrSS2",
    "dof_NacYaw", "dof_GenAz", "dof_DrTr",
    "dof_B1F1", "dof_B1E1", "dof_B1F2",
    "dof_B2F1", "dof_B2E1", "dof_B2F2",
    "dof_B3F1", "dof_B3E1", "dof_B3F2",
]

HYDRO_KEYS = [
    "WaveElev",
    "HydroFx", "HydroFy", "HydroFz",
    "HydroMx", "HydroMy", "HydroMz",
]

WIND_AERO_KEYS = [
    "WindSpeed",
    "AeroThrust",
    "AeroFx", "AeroFy", "AeroFz",
    "AeroMx", "AeroMy", "AeroMz",
    "GenTq",
    "GenPwr",
]

MOOR_KEYS = [
    "MoorFx", "MoorFy", "MoorFz",
    "MoorMx", "MoorMy", "MoorMz",
]

SAVE_GROUPS = {
    "Structure Response": DOF_KEYS,
    "Wave/Current Loads": HYDRO_KEYS,
    "Wind/Aero Loads": WIND_AERO_KEYS,
    "Mooring Loads": MOOR_KEYS,
}


def default_save_config():
    return {key: list(values) for key, values in SAVE_GROUPS.items()}

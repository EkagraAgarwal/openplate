"""Physical constants. Units: cm, s, mol, A, V, K, g (concentrations mol/cm3 unless the name says otherwise)."""
F = 96485.332          # C/mol (team slides used 98,456; wrong by 2 %)
R_GAS = 8.314462618    # J/(mol K)
M_CU = 63.546          # g/mol
RHO_CU = 8.96          # g/cm3
N_CU = 2               # electrons per Cu

# Cu thickness per charge: 1 mA/cm2 for 1 h -> 1.323 um at 100 % efficiency
UM_PER_MAH_CM2 = M_CU / (N_CU * F * RHO_CU) * 3.6 * 1e4


def thickness_um(j_mA_cm2: float, t_s: float, efficiency: float = 1.0) -> float:
    """Faraday thickness of a planar deposit."""
    return M_CU / (N_CU * F * RHO_CU) * j_mA_cm2 * 1e-3 * t_s * efficiency * 1e4


def celsius_to_K(T_C: float) -> float:
    return T_C + 273.15

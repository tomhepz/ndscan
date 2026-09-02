"""Registry of fit procedures provided by the optional OITG dependency."""

import oitg.fitting

FIT_OBJECTS: dict[str, oitg.fitting.FitBase.FitBase] = {
    name: getattr(oitg.fitting, name)
    for name in [
        "cos",
        "decaying_sinusoid",
        "detuned_square_pulse",
        "exponential_decay",
        "gaussian",
        "line",
        "lorentzian",
        "rabi_flop",
        "sinusoid",
        "v_function",
    ]
}
FIT_OBJECTS["parabola"] = oitg.fitting.shifted_parabola

__all__ = ["FIT_OBJECTS"]

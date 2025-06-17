# utils/aqi_utils.py
def calculate_aqi(cp, breakpoints):
    for bp_lo, bp_hi, i_lo, i_hi in breakpoints:
        if bp_lo <= cp <= bp_hi:
            return round(((i_hi - i_lo) / (bp_hi - bp_lo)) * (cp - bp_lo) + i_lo)
    return None
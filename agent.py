"""
Neural Nexus -- Round 2: The Trading Pit
agent.py

Locked Production Agent:
1. predict_market(): Anchored causal forecaster powered by regularized HuberRegressor
   models predicting next price movements using multi-scale momentum, mean reversion,
   and sector-pair divergence.
2. place_orders(): Crowd-aware live execution strategy using effective conviction
   (forecast move x crowding discount factor) with proportional lot allocation.
"""

import os
import joblib
import numpy as np

ASSETS = [
    "quantum_dynamics", "byte_stream", "gold_trust",
    "metro_rail", "agro_futures", "solar_grid"
]

SECTOR_PAIRS = {
    "quantum_dynamics": "byte_stream",
    "byte_stream": "quantum_dynamics",
    "gold_trust": "solar_grid",
    "solar_grid": "gold_trust",
    "metro_rail": "agro_futures",
    "agro_futures": "metro_rail"
}

BASELINE_PRICES = {
    "quantum_dynamics": 50.0,
    "byte_stream": 60.0,
    "gold_trust": 40.0,
    "metro_rail": 30.0,
    "agro_futures": 35.0,
    "solar_grid": 45.0
}

# Causal load relative to this file
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
try:
    BUNDLE = joblib.load(MODEL_PATH)
    MODELS = BUNDLE.get("models") if isinstance(BUNDLE, dict) else BUNDLE
except Exception:
    MODELS = None


def _clean_price_series(raw_prices, default_val):
    """Causal price sanitizer: strips NaNs and repairs recording spikes using past median."""
    if not raw_prices:
        return [float(default_val)]
    cleaned = []
    last_valid = float(default_val)
    for p in raw_prices:
        if p is not None and not np.isnan(p):
            last_valid = float(p)
            cleaned.append(last_valid)
        elif cleaned:
            cleaned.append(last_valid)

    if not cleaned:
        return [float(default_val)]

    # Causal running-median repair for 10x decimal errors
    if len(cleaned) >= 4:
        med = float(np.median(cleaned))
        if med > 1.0:
            repaired = []
            for v in cleaned:
                if v > 2.5 * med or v < 0.4 * med:
                    repaired.append(med)
                else:
                    repaired.append(v)
            return repaired
    return cleaned


def _extract_features(asset, cleaned_dict):
    """Extracts locked 6 causal features matching train_model.py."""
    p = np.array(cleaned_dict[asset], dtype=float)
    pair_a = SECTOR_PAIRS[asset]
    p_pair = np.array(cleaned_dict[pair_a], dtype=float)

    d1 = p[-1] - p[-2] if len(p) >= 2 else 0.0
    dev5 = p[-1] - np.mean(p[-5:]) if len(p) >= 5 else 0.0
    dev20 = p[-1] - np.mean(p[-20:]) if len(p) >= 20 else dev5

    pair_d1 = p_pair[-1] - p_pair[-2] if len(p_pair) >= 2 else 0.0
    pair_diff_d1 = d1 - pair_d1

    other_assets = [o for o in ASSETS if o != asset and o != pair_a]
    other_d1s = []
    for o in other_assets:
        po = cleaned_dict[o]
        if len(po) >= 2:
            other_d1s.append(po[-1] - po[-2])
    cross_d1 = float(np.mean(other_d1s)) if other_d1s else 0.0

    return [d1, dev5, dev20, pair_diff_d1, pair_d1, cross_d1]


def predict_market(recent_history):
    """
    recent_history: dict[asset_id -> list of up to last 20 realized prices]
    Returns dict[asset_id -> predicted price float for upcoming round].
    """
    cleaned_prices = {}
    for a in ASSETS:
        raw = recent_history.get(a, []) if isinstance(recent_history, dict) else []
        cleaned_prices[a] = _clean_price_series(raw, BASELINE_PRICES[a])

    predictions = {}
    for a in ASSETS:
        p = cleaned_prices[a]
        last_price = p[-1]

        if len(p) < 2:
            predictions[a] = float(last_price)
            continue

        if MODELS is not None and a in MODELS:
            feats = _extract_features(a, cleaned_prices)
            pred_move = float(MODELS[a].predict([feats])[0])
            # Conservative bound to prevent extreme moves
            pred_move = max(min(pred_move, 6.0), -6.0)
            predictions[a] = float(last_price + pred_move)
        else:
            # Robust causal fallback if model file is unreadable
            d1 = p[-1] - p[-2]
            dev5 = p[-1] - np.mean(p[-5:])
            fallback_move = -0.05 * d1 - 0.02 * dev5
            predictions[a] = float(last_price + fallback_move)

    return predictions


def place_orders(game_state, team_history):
    """
    game_state: round info, assets with recent_prices, recent_buy/sell_volume
    team_history: past rounds history
    Returns: dict[asset_id -> signed int lots] summing (abs) to <= trade_budget (14)
    """
    budget = int(game_state.get("trade_budget", 14))
    assets_data = game_state.get("assets", {})

    # 1. Clean price series
    recent_history = {}
    for a in ASSETS:
        raw_p = assets_data.get(a, {}).get("recent_prices", [])
        recent_history[a] = _clean_price_series(raw_p, BASELINE_PRICES[a])

    # 2. Get forecast prices
    forecast_prices = predict_market(recent_history)

    # 3. Calculate risk-adjusted conviction & apply crowding adjustment
    effective_conviction = {}
    for a in ASSETS:
        p = recent_history[a]
        last_p = p[-1]
        pair_a = SECTOR_PAIRS[a]
        pair_p = recent_history[pair_a]

        if len(p) < 2:
            effective_conviction[a] = 0.0
            continue

        forecast_move = forecast_prices[a] - last_p

        # Multi-factor signal: model move + trend persistence + reversion anchor
        diffs = [p[i] - p[i-1] for i in range(1, len(p))]
        w = [0.85 ** (len(diffs) - 1 - i) for i in range(len(diffs))]
        ewma_mom = (sum(wt * d for wt, d in zip(w, diffs)) / sum(w)) if diffs else 0.0
        reversion_pull = BASELINE_PRICES[a] - last_p

        # Pair divergence support: check if asset is lagging its partner
        dev20_a = p[-1] - np.mean(p[-20:])
        dev20_pair = pair_p[-1] - np.mean(pair_p[-20:])
        pair_reversion = dev20_pair - dev20_a  # pull towards partner

        raw_signal = 0.45 * forecast_move + 0.75 * ewma_mom + 0.20 * reversion_pull + 0.10 * pair_reversion

        # Volatility normalization (risk-adjusted conviction)
        recent_vol = np.std(p[-10:]) if len(p) >= 10 else 1.0
        risk_adj_signal = raw_signal / max(recent_vol, 0.5)

        # Crowding dilution discount for our chosen side
        buy_vols = assets_data.get(a, {}).get("recent_buy_volume", [])
        sell_vols = assets_data.get(a, {}).get("recent_sell_volume", [])

        if buy_vols and sell_vols:
            last_buy = buy_vols[-1] if buy_vols[-1] is not None else 0
            last_sell = sell_vols[-1] if sell_vols[-1] is not None else 0
            est_liquidity = max(BASELINE_PRICES[a] * 0.3 * 0.6, 5.0)

            # If our trade aligns with the crowded dominant side, reduce conviction proportionally
            if risk_adj_signal > 0 and last_buy > last_sell:
                dom_ratio = last_buy / est_liquidity
                if dom_ratio > 1.0:
                    discount = 1.0 / dom_ratio
                    risk_adj_signal *= (0.50 + 0.50 * discount)
            elif risk_adj_signal < 0 and last_sell > last_buy:
                dom_ratio = last_sell / est_liquidity
                if dom_ratio > 1.0:
                    discount = 1.0 / dom_ratio
                    risk_adj_signal *= (0.50 + 0.50 * discount)

        # Filter out weak/uncertain noise (sit out)
        if abs(risk_adj_signal) < 0.08:
            risk_adj_signal = 0.0

        effective_conviction[a] = risk_adj_signal

    # 4. Proportional Integer Allocation (Strictly <= dynamic runtime budget)
    total_conviction = sum(abs(effective_conviction[a]) for a in ASSETS)
    orders = {a: 0 for a in ASSETS}

    if total_conviction > 1e-4:
        ranked = sorted(ASSETS, key=lambda a: abs(effective_conviction[a]), reverse=True)
        rem_budget = budget

        for a in ranked:
            prop = abs(effective_conviction[a]) / total_conviction
            lots = min(int(round(prop * budget)), rem_budget)
            if lots > 0:
                orders[a] = int(np.sign(effective_conviction[a])) * lots
                rem_budget -= lots

        # Assign remaining chips to top conviction asset
        if rem_budget > 0 and ranked:
            top_asset = ranked[0]
            orders[top_asset] += int(np.sign(effective_conviction[top_asset])) * rem_budget

    # Safety clamp: guarantee sum(abs(lots)) <= runtime budget
    curr_exposure = sum(abs(v) for v in orders.values())
    if curr_exposure > budget:
        sorted_by_size = sorted(ASSETS, key=lambda a: abs(orders[a]))
        for a in sorted_by_size:
            excess = curr_exposure - budget
            if excess <= 0:
                break
            reduction = min(abs(orders[a]), excess)
            orders[a] -= int(np.sign(orders[a])) * reduction
            curr_exposure -= reduction

    # Cast to standard pure Python ints
    return {a: int(orders[a]) for a in ASSETS}


if __name__ == "__main__":
    fake_hist = {a: [BASELINE_PRICES[a] + i * 0.1 for i in range(20)] for a in ASSETS}
    p = predict_market(fake_hist)
    print("Self-test predict_market:", p)

    fake_state = {
        "round": 10, "total_rounds": 150, "trade_budget": 14,
        "assets": {a: {"ticker": a[:4].upper(),
                       "recent_prices": [BASELINE_PRICES[a] + i * 0.1 for i in range(20)],
                       "recent_buy_volume": [15] * 20,
                       "recent_sell_volume": [5] * 20} for a in ASSETS}
    }
    o = place_orders(fake_state, [])
    print("Self-test place_orders:", o)
    print("Total exposure:", sum(abs(v) for v in o.values()))

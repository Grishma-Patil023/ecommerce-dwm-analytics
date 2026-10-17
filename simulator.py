"""
simulator.py
E-Commerce Campaign & Pricing Simulator — Decision-Support Tool.
Simulates customer conversion probability, order value, basket recommendations, and net profit margin.
Supports both programmatic API and interactive CLI modes.
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

CATEGORY_MARGINS = {
    "Electronics": 0.28,
    "Apparel": 0.45,
    "Home & Kitchen": 0.38,
    "Beauty": 0.52,
    "Toys": 0.35,
    "Sports": 0.40,
    "Books": 0.30
}

AVG_UNIT_PRICES = {
    "Electronics": 349.00,
    "Apparel": 69.50,
    "Home & Kitchen": 189.00,
    "Beauty": 45.00,
    "Toys": 59.00,
    "Sports": 115.00,
    "Books": 24.50
}

CROSS_SELL_RULES = {
    "Electronics": ["Accessories", "Apparel", "Home & Kitchen"],
    "Apparel": ["Beauty", "Sports", "Shoes"],
    "Home & Kitchen": ["Electronics", "Appliances", "Books"],
    "Toys": ["Books", "Apparel"],
    "Sports": ["Apparel", "Electronics"],
    "Beauty": ["Apparel", "Personal Care"],
    "Books": ["Toys", "Home & Kitchen"]
}

def run_simulation(
    category: str = "Electronics",
    age_group: str = "Adults",
    country: str = "USA",
    discount_pct: float = 15.0,
    quantity: int = 2,
    payment_method: str = "Credit Card",
    customer_segment: str = "Champions (High Value)",
    *args,
    **kwargs
) -> Dict[str, Any]:
    """Runs a complete E-Commerce What-If Pricing & Campaign Scenario Simulation."""
    verbose = kwargs.get("verbose", False)
    if verbose:
        print("\n" + "=" * 65)
        print("  E-COMMERCE REVENUE & CAMPAIGN SIMULATOR -- DECISION SUPPORT")
        print("=" * 65)
        print(f"\n  Target Category:    {category}")
        print(f"  Customer Segment:   {customer_segment} ({age_group}, {country})")
        print(f"  Proposed Discount:  {discount_pct:.1f}%")
        print(f"  Order Quantity:     {quantity} units")
        print(f"  Payment Method:     {payment_method}")

    # 1. Base Financials
    base_price = AVG_UNIT_PRICES.get(category, 100.0)
    gross_order_value = round(base_price * quantity, 2)
    discount_amount = round(gross_order_value * (discount_pct / 100.0), 2)
    net_revenue = round(gross_order_value - discount_amount, 2)
    
    base_margin_rate = CATEGORY_MARGINS.get(category, 0.35)
    raw_profit = gross_order_value * base_margin_rate
    net_profit = round(raw_profit - discount_amount, 2)
    net_margin_pct = round((net_profit / net_revenue) * 100.0, 1) if net_revenue > 0 else 0.0

    # 2. Conversion & Fulfillment Risk Assessment
    base_fulfillment_prob = 0.88 if payment_method != "Cash on Delivery" else 0.62
    discount_boost = min(discount_pct * 0.005, 0.10)
    conversion_prob = min(0.98, round(0.70 + discount_boost + (0.05 if customer_segment.startswith("Champions") else 0.0), 3))
    fulfillment_prob = min(0.99, round(base_fulfillment_prob + (0.02 if payment_method == "Credit Card" else 0.0), 3))
    cancellation_risk = round(1.0 - fulfillment_prob, 3)

    # 3. Cross-Sell Recommendations (Apriori Basket Affinities)
    recommended_bundles = CROSS_SELL_RULES.get(category, ["Apparel", "Accessories"])

    # 4. Assessment Rating
    if net_profit <= 0:
        verdict = "[MARGIN-DILUTING] Proposed discount exceeds product gross margin rate."
    elif cancellation_risk > 0.30:
        verdict = "[ELEVATED RISK] Cash on Delivery exhibits significantly higher return/cancellation risk."
    elif net_margin_pct >= 20.0 and conversion_prob >= 0.80:
        verdict = "[HIGHLY RECOMMENDED] Profitable margins paired with high purchase conversion probability."
    else:
        verdict = "[VIABLE CAMPAIGN] Balanced revenue generation and customer retention incentives."

    result = {
        "scenario_inputs": {
            "category": category,
            "age_group": age_group,
            "country": country,
            "discount_pct": discount_pct,
            "quantity": quantity,
            "payment_method": payment_method,
            "customer_segment": customer_segment
        },
        "financial_projections": {
            "gross_order_value": gross_order_value,
            "discount_applied": discount_amount,
            "net_revenue": net_revenue,
            "net_profit": net_profit,
            "net_margin_percentage": net_margin_pct
        },
        "behavioral_probabilities": {
            "conversion_probability": conversion_prob,
            "fulfillment_probability": fulfillment_prob,
            "cancellation_risk": cancellation_risk
        },
        "cross_sell_recommendations": recommended_bundles,
        "recommendation": verdict
    }

    verbose = kwargs.get("verbose", False)
    if verbose:
        print("\nSimulated Financial Projections:")
        print(f"  Gross Order Value:   ${gross_order_value:>10,.2f}")
        print(f"  Discount Subsidized: -${discount_amount:>9,.2f} ({discount_pct:.1f}%)")
        print(f"  Net Realized Revenue: ${net_revenue:>9,.2f}")
        print(f"  Net Profit:           ${net_profit:>9,.2f} (Margin: {net_margin_pct:.1f}%)")

        print("\nBehavioral Likelihoods:")
        print(f"  Conversion Propensity: {conversion_prob*100:>6.1f}%")
        print(f"  Fulfillment Success:   {fulfillment_prob*100:>6.1f}%")
        print(f"  Cancellation Risk:     {cancellation_risk*100:>6.1f}%")

        print(f"\nRecommended Basket Cross-Sells: {', '.join(recommended_bundles)}")
        print(f"\nSTRATEGIC DECISION: {verdict}")

    out_file = BASE_DIR / "ml_results" / "simulation_result.json"
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    verbose = kwargs.get("verbose", False)
    if verbose:
        print(f"\n  ✓ Simulation results archived to: {out_file.name}")
    return result

def interactive_cli():
    print("\n" + "=" * 65)
    print("  E-COMMERCE CAMPAIGN SIMULATOR -- Interactive Mode")
    print("=" * 65)
    print("  (Press Enter for default values)")

    cat = input("  Category [Electronics]: ").strip() or "Electronics"
    seg = input("  Customer Segment [Champions (High Value)]: ").strip() or "Champions (High Value)"
    cntry = input("  Country [USA]: ").strip() or "USA"
    try:
        disc = float(input("  Discount % [15.0]: ").strip() or "15.0")
    except ValueError:
        disc = 15.0
    try:
        qty = int(input("  Quantity [2]: ").strip() or "2")
    except ValueError:
        qty = 2
    pay = input("  Payment Method [Credit Card]: ").strip() or "Credit Card"

    return run_simulation(
        category=cat, customer_segment=seg, country=cntry,
        discount_pct=disc, quantity=qty, payment_method=pay, verbose=True
    )

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_cli()
    else:
        run_simulation(verbose=True)

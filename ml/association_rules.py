"""
ml/association_rules.py — Market Basket Analysis using Apriori Algorithm.
Discovers high-lift cross-sell affinities between categories and products.
"""
import json
from pathlib import Path
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from .db_loader import load_orders_ml_df

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "ml_results"
OUTPUT_DIR.mkdir(exist_ok=True)

def run_association_rules(min_support=0.03, min_confidence=0.15):
    print("\n[ML] Running Market Basket Analysis (Apriori & Association Rules)...")
    df = load_orders_ml_df()
    
    # Customer transaction basket at category level
    basket = df.groupby(["customer_id", "category"])["quantity"].sum().unstack().fillna(0)
    basket_sets = (basket > 0).astype(bool)
    
    frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
    
    rules_list = []
    if not frequent_itemsets.empty:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
        rules = rules.sort_values("lift", ascending=False)
        
        for _, row in rules.head(15).iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])
            rules_list.append({
                "if_buys": antecedents,
                "then_recommends": consequents,
                "support": round(float(row["support"]), 4),
                "confidence": round(float(row["confidence"]), 4),
                "lift": round(float(row["lift"]), 4)
            })
            
    summary = {
        "min_support": min_support,
        "frequent_itemsets_found": len(frequent_itemsets),
        "total_rules_extracted": len(rules_list),
        "top_rules": rules_list
    }
    
    out_file = OUTPUT_DIR / "association_rules.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"  ✓ Found {len(frequent_itemsets)} itemsets and {len(rules_list)} high-lift cross-sell rules.")
    print(f"  ✓ Saved to: {out_file.name}")
    return summary

if __name__ == "__main__":
    run_association_rules()

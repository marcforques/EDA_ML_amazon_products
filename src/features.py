import pandas as pd
import numpy as np

def classify_sales(value):
    
    if pd.isna(value):
        return np.nan
    elif value <= 100:
        return "low"
    elif value <= 1000:
        return "medium"
    elif value <= 10000:
        return "high"
    else:
        return "very_high"
    
    
def get_model_features():
    
    return [
        "rating",
        "number_of_reviews",
        "current_discounted_price",
        "price_on_variant",
        "listed_price",
        "is_best_seller",
        "is_sponsored",
        "is_couponed",
        "buy_box_availability",
        "sustainability_badges",
        "coupon_discount_value",
        "discount_amount",
        "discount_percentage",
        "title_length",
        "title_word_count",
        "has_free_delivery",
        "has_fastest_delivery"
    ]
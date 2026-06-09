import re
import numpy as np
import pandas as pd


def parse_rating(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    match = re.search(r"(\d+(?:\.\d+)?)\s*out of", text)

    if match:
        return float(match.group(1))

    return np.nan


def parse_integer(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    match = re.search(r"\d[\d,]*", text)

    if match:
        return int(match.group(0).replace(",", ""))

    return np.nan


def parse_price(value, require_dollar=False):
    if pd.isna(value):
        return np.nan

    text = str(value)

    if require_dollar and "$" not in text:
        return np.nan

    text = text.replace(",", "")
    match = re.search(r"\$?\s*(\d+(?:\.\d+)?)", text)

    if match:
        return float(match.group(1))

    return np.nan


def parse_bought_last_month(value):
    if pd.isna(value):
        return np.nan

    text = str(value)
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*([Kk]?)\+\s*bought in past month",
        text
    )

    if not match:
        return np.nan

    number = float(match.group(1))
    multiplier = 1000 if match.group(2).lower() == "k" else 1

    return int(number * multiplier)


def parse_coupon_value(value):
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text == "no coupon":
        return np.nan

    text = text.replace(",", "")
    match = re.search(r"save\s+\$?(\d+(?:\.\d+)?)\s*(%)?", text)

    if match:
        return float(match.group(1))

    return np.nan


def parse_coupon_type(value):
    if pd.isna(value):
        return "no_coupon"

    text = str(value).strip().lower()

    if text == "no coupon":
        return "no_coupon"

    if "%" in text:
        return "percentage"

    if "$" in text:
        return "amount"

    return "other"


# Función de limpieza completa
def clean_dataset(df):
    df_clean = df.copy()

    df_clean = df_clean.rename(columns={"current/discounted_price": "current_discounted_price"})

    df_clean["rating"] = df["rating"].apply(parse_rating)
    df_clean["number_of_reviews"] = df["number_of_reviews"].apply(parse_integer)
    df_clean["bought_in_last_month"] = df["bought_in_last_month"].apply(parse_bought_last_month)

    df_clean["current_discounted_price"] = df["current/discounted_price"].apply(parse_price)
    df_clean["price_on_variant"] = df["price_on_variant"].apply(lambda x: parse_price(x, require_dollar=True))
    df_clean["listed_price"] = df["listed_price"].apply(parse_price)

    df_clean["coupon_discount_value"] = df["is_couponed"].apply(parse_coupon_value)
    df_clean["coupon_discount_type"] = df["is_couponed"].apply(parse_coupon_type)

    coupon_text = df["is_couponed"].fillna("").astype(str).str.strip().str.lower()
    df_clean["is_couponed"] = coupon_text.str.contains("with coupon").astype(int)

    df_clean["is_best_seller"] = (df["is_best_seller"] == "Best Seller").astype(int)

    df_clean["is_sponsored"] = (df["is_sponsored"] == "Sponsored").astype(int)

    df_clean["buy_box_availability"] = (df["buy_box_availability"] == "Add to cart").astype(int)

    df_clean["sustainability_badges"] = (df["sustainability_badges"].notna()).astype(int)

    delivery_text = df["delivery_details"].fillna("").astype(str).str.lower()

    df_clean["has_free_delivery"] = (delivery_text.str.contains("free delivery")).astype(int)

    df_clean["has_fastest_delivery"] = (delivery_text.str.contains("fastest delivery")).astype(int)

    df_clean["collected_at"] = pd.to_datetime(df["collected_at"],errors="coerce")

    df_clean["discount_amount"] = (df_clean["listed_price"] - df_clean["current_discounted_price"])

    df_clean.loc[df_clean["discount_amount"] < 0,"discount_amount"] = np.nan

    df_clean["discount_percentage"] = (df_clean["discount_amount"] / df_clean["listed_price"] * 100)

    df_clean["title_length"] = df["title"].astype(str).str.len()
    df_clean["title_word_count"] = df["title"].astype(str).str.split().str.len()

    columns_to_drop = [
        "product_url",
        "image_url",
        "delivery_details"
    ]

    df_clean = df_clean.drop(columns=columns_to_drop, errors="ignore")

    return df_clean

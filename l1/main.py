"""
L1 Project - Exercises: Customer Transactions Analysis
======================================================
Learning Objectives:
    Access, clean, transform, merge and aggregate data to report and visualize the results.
"""

import os
import sqlite3
import warnings

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

INPUT_DIR  = os.path.join(os.path.dirname(__file__), "input")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# STEP 1 – DATA ACCESS
# ==============================================================================

def step1_data_access():
    print("\n" + "="*70)
    print("STEP 1 – DATA ACCESS")
    print("="*70)

    customer1 = pd.read_excel(os.path.join(INPUT_DIR, "CustomerInfoSystem1.xlsx"))
    print(f"  [Excel]   CustomerInfoSystem1   → {customer1.shape}")

    customer2 = pd.read_excel(os.path.join(INPUT_DIR, "CustomerInfoSystem2.xlsx"))
    print(f"  [Excel]   CustomerInfoSystem2   → {customer2.shape}")

    stores = pd.read_csv(os.path.join(INPUT_DIR, "Stores.csv"))
    print(f"  [CSV]     Stores                → {stores.shape}")

    conn            = sqlite3.connect(os.path.join(INPUT_DIR, "Sales.sqlite"))
    transactions    = pd.read_sql("SELECT * FROM Transactions",      conn)
    product_min_max = pd.read_sql("SELECT * FROM ProductNrAndPrice", conn)
    conn.close()

    print(f"  [SQLite]  Transactions          → {transactions.shape}")
    print(f"  [SQLite]  ProductNrAndPrice     → {product_min_max.shape}")

    return customer1, customer2, stores, transactions, product_min_max


# ==============================================================================
# STEP 2 – DATA CLEANING
# ==============================================================================

def table_cropper(df, crop_rows=0, crop_cols=0):
    df = df.iloc[crop_rows:, crop_cols:].copy()
    df.columns = df.columns.str.strip()
    return df.reset_index(drop=True)


def duplicate_row_filter(df, col, keep="first"):
    before = len(df)
    df = df.drop_duplicates(subset=[col], keep=keep)
    print(f"    [Dup Filter]  '{col}': removed {before - len(df)} duplicate(s)")
    return df


def missing_value_imputer(df, col, strategy="mean"):
    df = df.copy()
    df[col] = pd.to_numeric(df[col], errors="coerce")
    n_missing = df[col].isna().sum()
    if n_missing == 0:
        return df
    if strategy == "mean":
        fill_val = df[col].mean()
    elif strategy == "median":
        fill_val = df[col].median()
    elif strategy == "mode":
        fill_val = df[col].mode()[0]
    else:
        fill_val = strategy
    df[col] = df[col].fillna(round(float(fill_val), 2))
    print(f"    [Missing Val] '{col}': filled {n_missing} value(s) with {strategy}={fill_val:.2f}")
    return df


def step2_data_cleaning(customer1, customer2):
    print("\n" + "="*70)
    print("STEP 2 – DATA CLEANING")
    print("="*70)

    print("  > CustomerInfoSystem1")
    df1 = table_cropper(customer1, crop_rows=1, crop_cols=1)
    df1 = duplicate_row_filter(df1, col="CustomerID")
    df1 = missing_value_imputer(df1, col="Age", strategy="mean")
    print(f"    → cleaned shape: {df1.shape}")

    print("  > CustomerInfoSystem2")
    df2 = table_cropper(customer2, crop_rows=0, crop_cols=0)
    df2 = duplicate_row_filter(df2, col="CustomerID")
    df2 = missing_value_imputer(df2, col="Age", strategy="mean")
    print(f"    → cleaned shape: {df2.shape}")

    return df1, df2


# ==============================================================================
# STEP 3 – DATA TRANSFORMATION
# ==============================================================================

def step3_data_transformation(df1, df2):
    print("\n" + "="*70)
    print("STEP 3 – DATA TRANSFORMATION")
    print("="*70)

    def age_group(age):
        try:
            age = float(age)
        except (TypeError, ValueError):
            return None
        if age < 18:
            return "Adolescent"
        elif age <= 65:
            return "Adult"
        else:
            return "Older Adult"

    for df, name in [(df1, "CIS1"), (df2, "CIS2")]:
        # AgeGroup
        if "Age" in df.columns:
            df["AgeGroup"] = df["Age"].apply(age_group)
            print(f"  [{name}] AgeGroup → {df['AgeGroup'].value_counts().to_dict()}")

        # Country: replace "," with " "
        if "Country" in df.columns:
            df["Country"] = df["Country"].astype(str).str.replace(",", " ", regex=False)

        # Cell Splitter – CustomerID → CustomerGroup, CustomerID
        if "CustomerID" in df.columns:
            split = df["CustomerID"].astype(str).str.split("_", n=1, expand=True)
            df["CustomerGroup"] = split[0]
            df["CustomerID"]    = split[1]
            print(f"  [{name}] CustomerID split → CustomerGroup, CustomerID")

        # Column Merger – Email เป็นหลัก fallback CorporateEmail
        if "Email" in df.columns and "CorporateEmail" in df.columns:
            df["Email"] = df["Email"].where(df["Email"].notna(), df["CorporateEmail"])
            print(f"  [{name}] Email: used Email where available, fallback CorporateEmail")

        # Number to String – Newsletter
        if "Newsletter" in df.columns:
            df["Newsletter"] = df["Newsletter"].astype(str)
            print(f"  [{name}] Newsletter converted to string")

    df1.to_excel(os.path.join(OUTPUT_DIR, "S3_CustomerInfoSystem1_transformed.xlsx"), index=False)
    df2.to_excel(os.path.join(OUTPUT_DIR, "S3_CustomerInfoSystem2_transformed.xlsx"), index=False)
    print(f"  [Export] S3_CustomerInfoSystem1_transformed.xlsx → {df1.shape}")
    print(f"  [Export] S3_CustomerInfoSystem2_transformed.xlsx → {df2.shape}")

    return df1, df2


# ==============================================================================
# STEP 4 – DATA MERGING
# ==============================================================================

def step4_data_merging(df1, df2, stores, transactions, product_min_max):
    print("\n" + "="*70)
    print("STEP 4 – DATA MERGING")
    print("="*70)

    # Concatenate
    customers = pd.concat([df1, df2], ignore_index=True)
    print(f"  [Concatenate]  customers → {customers.shape}")

    # Value Lookup – StoreType from Stores
    if "StoreID" in transactions.columns and "StoreID" in stores.columns:
        transactions["StoreType"] = transactions["StoreID"].map(
            stores.set_index("StoreID")["StoreType"]
        )
        print("  [Value Lookup] StoreType appended")

    # Value Lookup – Price from ProductNrAndPrice
    price_key = next(
        (c for c in ["ProductNr", "ProductID", "Product_ID"]
         if c in transactions.columns and c in product_min_max.columns),
        None
    )
    if price_key:
        transactions["Price"] = transactions[price_key].map(
            product_min_max.set_index(price_key)["Price"]
        )
        print(f"  [Value Lookup] Price appended via '{price_key}'")

    # Joiner – inner join on CustomerID
    merged = pd.merge(customers, transactions, on="CustomerID", how="inner",
                      suffixes=("_cust", "_trans"))
    print(f"  [Joiner]       merged → {merged.shape}")

    merged.to_csv(os.path.join(OUTPUT_DIR, "S4_merged.csv"), index=False)
    customers.to_csv(os.path.join(OUTPUT_DIR, "S4_customers.csv"), index=False)
    transactions.to_csv(os.path.join(OUTPUT_DIR, "S4_transactions.csv"), index=False)

    return customers, transactions, merged


# ==============================================================================
# STEP 5 – DATA AGGREGATION
# ==============================================================================

def step5_data_aggregation(merged):
    print("\n" + "="*70)
    print("STEP 5 – DATA AGGREGATION")
    print("="*70)

    price_col   = "Price"
    order_col   = "OrderNumber"
    product_col = "ProductNr"

    # 5.1 – Total Price per Customer
    agg1 = (
        merged.groupby("CustomerID", as_index=False)[price_col]
        .sum()
        .rename(columns={price_col: "TotalPrice_per_Customer"})
    )
    print(f"  [5.1] TotalPrice per Customer → {agg1.shape}")

    # 5.2 – BasketSize pivot
    basket = (
        merged.groupby([order_col, "StoreType"], as_index=False)[product_col]
        .count()
        .rename(columns={product_col: "BasketSize"})
    )
    pivot_basket = basket.pivot_table(
        index="BasketSize", columns="StoreType",
        values=order_col, aggfunc="count", fill_value=0
    )
    print(f"  [5.2] BasketSize pivot → {pivot_basket.shape}")

    # 5.3 – Detailed per-customer GroupBy (Unique Count ProductNr)
    agg3 = (
        merged.groupby(["CustomerID", "CustomerGroup", "AgeGroup"], as_index=False)
        .agg(
            Sum_Price        = ("Price",     "sum"),
            Unique_ProductNr = ("ProductNr", "nunique")
        )
    )
    print(f"  [5.3] Customer aggregation → {agg3.shape}")

    agg1.to_csv(os.path.join(OUTPUT_DIR, "S5_1_TotalPrice_per_Customer.csv"), index=False)
    pivot_basket.to_csv(os.path.join(OUTPUT_DIR, "S5_2_BasketSize_Pivot.csv"))
    agg3.to_csv(os.path.join(OUTPUT_DIR, "S5_3_Customer_Aggregation.csv"), index=False)

    return agg1, pivot_basket, agg3


# ==============================================================================
# STEP 6 – EXPORT, VISUALIZATION & REPORTING
# ==============================================================================

def step6_export_visualize_report(agg1, pivot_basket, agg3, merged):
    print("\n" + "="*70)
    print("STEP 6 – EXPORT, VISUALIZATION & REPORTING")
    print("="*70)

    colors = plt.cm.tab10.colors

    # 6.1 – Total Price per Customer → Excel
    excel_out = os.path.join(OUTPUT_DIR, "S6_1_TotalPrice_per_Customer.xlsx")
    agg1.to_excel(excel_out, index=False)
    print(f"  [Excel Writer] → {excel_out}")

    # 6.2 – Bar Chart: BasketSize per StoreType → PNG
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot_basket.plot(kind="bar", ax=ax, color=["#5B7BE8", "#7BC67E"],
                      edgecolor="white", width=0.7)
    ax.set_title("Basket Size Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Basket Size", fontsize=11)
    ax.set_ylabel("Number of orders", fontsize=11)
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    png_out = os.path.join(OUTPUT_DIR, "S6_2_BasketSize_BarChart.png")
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Bar Chart PNG] → {png_out}")

    # 6.3 – PDF Report
    pdf_path = os.path.join(OUTPUT_DIR, "S6_3_Report.pdf")
    with PdfPages(pdf_path) as pdf:

        # 3.4 Title page
        fig, ax = plt.subplots(figsize=(11.7, 8.3))
        ax.axis("off")
        ax.text(0.5, 0.60, "Customer Transactions Analysis",
                ha="center", va="center", fontsize=32, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.48, "L1 Project – Exercises",
                ha="center", va="center", fontsize=18, color="#555", transform=ax.transAxes)
        ax.text(0.5, 0.38, "Monthly Insights Report",
                ha="center", va="center", fontsize=14, color="#888", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  [PDF] Title page")

        # 3.1 Scatter Plot: Sum(Price) vs Unique Count(ProductNr)
        fig, ax = plt.subplots(figsize=(11, 7))
        for i, grp in enumerate(sorted(agg3["CustomerGroup"].unique())):
            sub = agg3[agg3["CustomerGroup"] == grp]
            ax.scatter(sub["Unique_ProductNr"], sub["Sum_Price"],
                       color=colors[i % len(colors)], alpha=0.7, s=60, label=grp)
        ax.set_title("Sum of Price vs Unique Count of ProductNr",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Unique Count (ProductNr)", fontsize=11)
        ax.set_ylabel("Sum of Price", fontsize=11)
        ax.legend(title="Customer Group", fontsize=9)
        ax.grid(linestyle="--", alpha=0.4)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  [PDF] Scatter Plot")

        # 3.2 Bar Chart: Count of occurrences per CustomerGroup
        fig, ax = plt.subplots(figsize=(10, 6))
        grp_count = agg3["CustomerGroup"].value_counts().sort_index()
        grp_count.plot(kind="bar", ax=ax,
                       color=[colors[i % len(colors)] for i in range(len(grp_count))],
                       edgecolor="white", width=0.6)
        ax.set_title("Count of Occurrences per Customer Group",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Customer Group", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  [PDF] CustomerGroup Bar Chart")

        # 3.3 Parallel Coordinates: AgeGroup, CustomerGroup, Sum(Price), Unique Count(ProductNr)
        axes_data   = ["AgeGroup", "CustomerGroup", "Sum_Price", "Unique_ProductNr"]
        axes_labels = ["AgeGroup", "CustomerGroup", "Sum(Price)", "Unique Count\n(ProductNr)"]

        plot_df = agg3[["CustomerGroup"] + axes_data].dropna().copy()

        age_order = ["Adolescent", "Adult", "Older Adult"]
        grp_order = sorted(agg3["CustomerGroup"].unique())
        plot_df["AgeGroup"]     = plot_df["AgeGroup"].map({v: i for i, v in enumerate(age_order)})
        plot_df["CustomerGroup_num"] = plot_df["CustomerGroup"].map({v: i for i, v in enumerate(grp_order)})

        num_cols = ["AgeGroup", "CustomerGroup_num", "Sum_Price", "Unique_ProductNr"]
        norm_df  = plot_df[num_cols].copy()
        for c in num_cols:
            cmin, cmax = norm_df[c].min(), norm_df[c].max()
            norm_df[c] = (norm_df[c] - cmin) / (cmax - cmin + 1e-9)

        fig, ax = plt.subplots(figsize=(12, 7))
        for i, grp in enumerate(grp_order):
            idx = plot_df["CustomerGroup_num"] == i
            for _, row in norm_df[idx].iterrows():
                ax.plot(range(len(num_cols)), row[num_cols].values,
                        color=colors[i % len(colors)], alpha=0.2, linewidth=0.8)
            median_row = norm_df[idx][num_cols].median()
            ax.plot(range(len(num_cols)), median_row.values,
                    color=colors[i % len(colors)], linewidth=2.5, label=grp)

        ax.set_xticks(range(len(num_cols)))
        ax.set_xticklabels(axes_labels, fontsize=10)
        ax.set_ylabel("Normalised Value", fontsize=10)
        ax.set_title("Parallel Coordinates Plot", fontsize=14, fontweight="bold")
        ax.legend(title="Customer Group", bbox_to_anchor=(1.01, 1),
                  loc="upper left", fontsize=9)
        ax.grid(axis="x", linestyle="--", alpha=0.3)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)
        print("  [PDF] Parallel Coordinates")

    print(f"  [PDF Writer] → {pdf_path}")

    # Intermediate CSV exports
    agg1.to_csv(os.path.join(OUTPUT_DIR, "S5_1_TotalPrice_per_Customer.csv"), index=False)
    pivot_basket.to_csv(os.path.join(OUTPUT_DIR, "S5_2_BasketSize_Pivot.csv"))
    agg3.to_csv(os.path.join(OUTPUT_DIR, "S5_3_Customer_Aggregation.csv"), index=False)
    print("  [CSV exports] Intermediate files saved")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "#"*70)
    print("#  L1 PROJECT – CUSTOMER TRANSACTIONS ANALYSIS")
    print("#"*70)

    customer1, customer2, stores, transactions, product_min_max = step1_data_access()
    df1, df2                        = step2_data_cleaning(customer1, customer2)
    df1, df2                        = step3_data_transformation(df1, df2)
    customers, transactions, merged = step4_data_merging(df1, df2, stores, transactions, product_min_max)
    agg1, pivot_basket, agg3        = step5_data_aggregation(merged)
    step6_export_visualize_report(agg1, pivot_basket, agg3, merged)

    print("\n" + "#"*70)
    print("#  WORKFLOW COMPLETE – check the output/ folder")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()

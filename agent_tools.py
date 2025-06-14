import pandas as pd
from smolagents import tool

@tool
def total_revenue(df: pd.DataFrame) -> float:
    """
    Calculates the total revenue from the sales log.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The total revenue as a float.
    """
    return df['Total Price'].sum()

@tool
def total_cost(df: pd.DataFrame) -> float:
    """
    Calculates the total cost from the sales log.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The total cost as a float.
    """
    return (df['Cost per Unit'] * df['Quantity']).sum()

@tool
def total_profit(df: pd.DataFrame) -> float:
    """
    Calculates the total profit from the sales log.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The total profit as a float.
    """
    return df['Profit'].sum()

@tool
def gross_margin(df: pd.DataFrame) -> float:
    """
    Calculates the gross margin percentage.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The gross margin as a percentage (float).
    """
    revenue = df['Total Price'].sum()
    profit = df['Profit'].sum()
    return (profit / revenue) * 100 if revenue else 0.0

@tool
def total_units_sold(df: pd.DataFrame) -> int:
    """
    Calculates the total number of units sold.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The total units sold as an integer.
    """
    return df['Quantity'].sum()

@tool
def number_of_transactions(df: pd.DataFrame) -> int:
    """
    Calculates the number of transactions.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The number of transactions (rows) as an integer.
    """
    return len(df)

@tool
def average_basket_size(df: pd.DataFrame) -> float:
    """
    Calculates the average basket size (units per transaction).

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The average basket size as a float.
    """
    return df['Quantity'].sum() / len(df) if len(df) else 0.0

@tool
def peak_sales_hour(df: pd.DataFrame) -> str:
    """
    Finds the time window with the highest revenue.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The time (as a string) with the highest total revenue.
    """
    peak = df.groupby('Time')['Total Price'].sum().idxmax()
    return str(peak)

@tool
def top_selling_item(df: pd.DataFrame) -> str:
    """
    Finds the top-selling item by quantity.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The name of the top-selling item.
    """
    return df.groupby('Item')['Quantity'].sum().idxmax()

@tool
def most_profitable_item(df: pd.DataFrame) -> str:
    """
    Finds the most profitable item by total profit.

    Args:
        df: The sales log as a pandas DataFrame.

    Returns:
        The name of the most profitable item.
    """
    return df.groupby('Item')['Profit'].sum().idxmax()

@tool
def low_stock_items(inventory_df: pd.DataFrame, threshold: int = 10) -> list:
    """
    Finds items with stock below a given threshold.

    Args:
        inventory_df: The inventory DataFrame with columns 'Item' and 'Stock'.
        threshold: The stock threshold for low stock.

    Returns:
        A list of item names with stock below the threshold.
    """
    return inventory_df[inventory_df['Stock'] < threshold]['Item'].tolist()

@tool
def unsold_items(inventory_df: pd.DataFrame, sales_df: pd.DataFrame) -> list:
    """
    Finds items in inventory that had zero sales today.

    Args:
        inventory_df: The inventory DataFrame with 'Item'.
        sales_df: The sales log DataFrame with 'Item'.

    Returns:
        A list of unsold item names.
    """
    sold_items = set(sales_df['Item'])
    all_items = set(inventory_df['Item'])
    return list(all_items - sold_items)

@tool
def fast_movers(inventory_df: pd.DataFrame) -> list:
    """
    Finds items that are sold out or nearly sold out.

    Args:
        inventory_df: The inventory DataFrame with columns 'Item' and 'Stock'.

    Returns:
        A list of item names with stock <= 1.
    """
    return inventory_df[inventory_df['Stock'] <= 1]['Item'].tolist()

@tool
def low_margin_high_sellers(df: pd.DataFrame, margin_threshold: float = 10.0, qty_threshold: int = 5) -> list:
    """
    Flags high-selling items with low profit margins.

    Args:
        df: The sales log as a pandas DataFrame.
        margin_threshold: The maximum margin (%) to flag as low.
        qty_threshold: The minimum quantity sold to consider as high-selling.

    Returns:
        A list of item names that are high-selling but low-margin.
    """
    grouped = df.groupby('Item').agg({'Quantity': 'sum', 'Total Price': 'sum', 'Profit': 'sum'})
    grouped['Margin'] = (grouped['Profit'] / grouped['Total Price']) * 100
    flagged = grouped[(grouped['Quantity'] >= qty_threshold) & (grouped['Margin'] < margin_threshold)]
    return flagged.index.tolist()

@tool
def natural_language_summary(
    total_revenue: float,
    total_cost: float,
    total_profit: float,
    gross_margin: float,
    total_units_sold: int,
    number_of_transactions: int,
    average_basket_size: float,
    peak_sales_hour: str,
    top_selling_item: str,
    most_profitable_item: str,
    low_stock_items: list = None,
    unsold_items: list = None,
    fast_movers: list = None,
    low_margin_high_sellers: list = None
) -> str:
    """
    Generates a natural language summary of the daily sales and inventory report.

    Args:
        total_revenue: Total revenue for the day.
        total_cost: Total cost for the day.
        total_profit: Total profit for the day.
        gross_margin: Gross margin percentage.
        total_units_sold: Total units sold.
        number_of_transactions: Number of transactions.
        average_basket_size: Average basket size.
        peak_sales_hour: Time window with highest revenue.
        top_selling_item: Top-selling item by quantity.
        most_profitable_item: Most profitable item by profit.
        low_stock_items: List of low stock items.
        unsold_items: List of unsold items.
        fast_movers: List of fast-moving (sold out/nearly sold out) items.
        low_margin_high_sellers: List of high-selling, low-margin items.

    Returns:
        A human-readable summary string.
    """
    summary = (
        f"Your store earned ${total_revenue:,.2f} in revenue and made ${total_profit:,.2f} in profit today, "
        f"with a gross margin of {gross_margin:.1f}%. "
        f"A total of {total_units_sold} items were sold in {number_of_transactions} transactions. "
        f"Average basket size was {average_basket_size:.2f} units per transaction. "
        f"Peak sales occurred at {peak_sales_hour}. "
        f"{top_selling_item} led sales, while {most_profitable_item} was the most profitable item. "
    )
    if low_stock_items:
        summary += f"\nLow stock items: {', '.join(low_stock_items)}."
    if unsold_items:
        summary += f"\nUnsold items: {', '.join(unsold_items)}."
    if fast_movers:
        summary += f"\nFast movers (sold out/nearly sold out): {', '.join(fast_movers)}."
    if low_margin_high_sellers:
        summary += f"\nHigh-selling, low-margin items: {', '.join(low_margin_high_sellers)}."
    return summary 
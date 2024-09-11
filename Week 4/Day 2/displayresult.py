import pandas as pd
from IPython.display import HTML


# Function to generate arrows based on comparison between two columns
def get_arrow(value1, value2):
    if value2 > value1:
        return '↑'  # up arrow (improvement)
    elif value2 < value1:
        return '↓'  # down arrow (decrease)
    else:
        return '='   # no change

# Prepare DataFrame for display, including arrows in HTML
def highlight_improvement(df, columns):
    # Iterate over the columns in pairs to compare
    for i in range(1, len(columns)):
        prev_col = columns[i - 1]
        curr_col = columns[i]
        
        # Ensure we are comparing numeric values and then adding arrows
        df[curr_col] = df.apply(lambda row: f"{row[curr_col]} {get_arrow(row[prev_col], row[curr_col])}", axis=1)

    # Define the function to color the arrows
    def color_arrow(val):
        green_arrow = "<span style='color:green'>↑</span>"
        red_arrow = "<span style='color:red'>↓</span>"
        if '↑' in val:
            return f"{val.replace('↑', green_arrow)}"
        elif '↓' in val:
            return f"{val.replace('↓', red_arrow)}"
        else:
            return val

    # Apply color formatting to all relevant columns
    df_arrows = df.copy()
    for col in columns[1:]:  # Skip the first column since it doesn't need arrows
        df_arrows[col] = df_arrows[col].apply(color_arrow)
    
    return df_arrows

# Apply arrow highlighting and display
# df_with_arrows = highlight_improvement(df[['Metric', 'ADA', 'TE3', 'TE3 Compressed', 'TE3 MultiQuery']])
# HTML(df_with_arrows.to_html(escape=False, index=False))
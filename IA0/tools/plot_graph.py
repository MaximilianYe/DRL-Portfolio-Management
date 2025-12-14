import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import glob
from pathlib import Path
import matplotlib.dates as mdates
warnings.filterwarnings('ignore')

class IndustryStockAnalyzer:
    def __init__(self, data_folder='data/processed/train'):
        """
        Industry-based stock analyzer
        
        Parameters:
        data_folder: Stock data folder path
        """
        self.data_folder = data_folder
        self.stock_data = {}
        self.industry_mapping = self._create_industry_mapping()
        
    def _create_industry_mapping(self):
        """Create stock to industry mapping based on your data acquisition script classification"""
        industry_mapping = {
            # === Semiconductors/Electronics ===
            "Semiconductors/Electronics": [
                "SOXX", "SMH", "SOXL", "TSM", "INTC", "QCOM", 
                "MU", "AVGO", "TXN"
            ],
            
            # === Communications/Telecom ===
            "Communications/Telecom": [
                "XLC", "VOX", "FCOM", "T", "VZ", "TMUS", 
                "CMCSA", "DIS"
            ],
            
            # === Electric Vehicles/Clean Energy ===
            "Electric Vehicles/Clean Energy": [
                "TSLA", "NIO", "XPEV", "LI", "BYD"
            ],
            
            # === Traditional Finance ===
            "Traditional Finance": [
                "JPM", "BAC", "WFC", "GS", "MS", "V", "MA"
            ],
            
            # === Healthcare ===
            "Healthcare": [
                "JNJ", "UNH", "PFE", "MRNA", "ABBV"
            ],
            
            # === Consumer Goods ===
            "Consumer Goods": [
                "KO", "PEP", "PG", "WMT", "COST", "MCD", "SBUX"
            ],
            
            # === Energy ===
            "Energy": [
                "XOM", "CVX", "COP"
            ],
            
            # === Industrials ===
            "Industrials": [
                "BA", "CAT", "GE", "LMT", "BSX"
            ],
            
            # === Real Estate/REITs ===
            "Real Estate/REITs": [
                "AMT", "PLD", "CCI"
            ],
            
            # === Broad Market ETFs ===
            "Broad Market ETFs": [
                "SPY", "QQQ", "VTI", "IWM", "DIA"
            ],
            
            # === Sector ETFs ===
            "Sector ETFs": [
                "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", 
                "XLP", "XLRE", "XLU"
            ],
            
            # === Thematic ETFs ===
            "Thematic ETFs": [
                "ARKK", "ICLN", "FINX", "HACK", "ROBO"
            ],
            
            # === Gold/Precious Metals ===
            "Gold/Precious Metals": [
                "GLD", "IAU", "GDXJ", "SLV", "PPLT"
            ],
            
            # === Commodities ===
            "Commodities": [
                "DBC", "USO", "UNG"
            ],
            
            # === Bonds/Fixed Income ===
            "Bonds/Fixed Income": [
                "TLT", "IEF", "LQD", "HYG", "TIP"
            ],
            
            # === International Markets ===
            "International Markets": [
                "VEA", "VWO", "EFA", "FXI", "EWJ"
            ],

            "Internet/Technology": [
                # US Internet Giants
                "GOOGL", "GOOG", "META", "AMZN", "AAPL", "MSFT",
                # Cloud/Enterprise Software
                "CRM", "ADBE", "NOW", "WDAY", "ZM", "DDOG", "SNOW", "ORCL",
                # Social Media & Digital Platforms
                "SNAP", "PINS", "UBER", "LYFT",
                # E-commerce & Fintech
                "SHOP", "PYPL", "SQ",
                # Chinese Internet (US-listed)
                "BABA", "JD", "PDD", "BIDU", "VIPS", "DIDI", "BEKE",
                # Chinese Tech
                "NTES", "WB"
            ],
            
            # === Cryptocurrency Related ===
            "Cryptocurrency Related": [
                "COIN", "MSTR"
            ]
        }
        
        # Create reverse mapping: stock code -> industry
        reverse_mapping = {}
        for industry, stocks in industry_mapping.items():
            for stock in stocks:
                reverse_mapping[stock] = industry
        
        return reverse_mapping

    def load_stock_data(self):
        """Load all stock data"""
        csv_files = [f for f in os.listdir(self.data_folder) if f.endswith('.csv')]
        
        print(f"Found {len(csv_files)} data files")
        
        for csv_file in csv_files:
            try:
                file_path = os.path.join(self.data_folder, csv_file)
                df = pd.read_csv(file_path)
                
                # Try to parse date column
                date_col = None
                for col in df.columns:
                    if '日期' in col or 'date' in col.lower() or col.lower() == 'date':
                        date_col = col
                        break
                
                if date_col is None:
                    date_col = df.columns[0]  # Default first column is date
                
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
                
                # Extract stock code
                stock_name = csv_file.replace('.csv', '').replace('_stock_processed', '').replace('_processed', '')
                
                # Find price column and calculate cumulative return
                price_col = None
                for col in ['收盘', 'Close', 'close', '收盘价', 'close_price']:
                    if col in df.columns:
                        price_col = col
                        break
                
                if price_col is None:
                    print(f"Warning: {stock_name} no closing price column found, available columns: {list(df.columns)}")
                    continue
                
                # Calculate cumulative return
                df['Cumulative Return'] = (df[price_col] / df[price_col].iloc[0] - 1) * 100
                
                self.stock_data[stock_name] = df
                
            except Exception as e:
                print(f"Failed to read {csv_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.stock_data)} stock data files")
        return len(self.stock_data)
    
    def group_by_industry(self):
        """Group stock data by industry"""
        industry_groups = {}
        unclassified = {}
        
        for stock_name, df in self.stock_data.items():
            # Try to match industry
            industry = self.industry_mapping.get(stock_name, None)
            
            if industry:
                if industry not in industry_groups:
                    industry_groups[industry] = {}
                industry_groups[industry][stock_name] = df
            else:
                # Unclassified stocks
                unclassified[stock_name] = df
        
        # If there are unclassified stocks, add them to "Others" category
        if unclassified:
            industry_groups["Others"] = unclassified
        
        return industry_groups
    
    def _calculate_y_limits(self, industry_data, percentile_clip=5):
        """Calculate appropriate Y-axis range to avoid extreme values"""
        all_returns = []
        
        for stock_name, df in industry_data.items():
            if 'Cumulative Return' in df.columns:
                returns = df['Cumulative Return'].dropna()
                all_returns.extend(returns.tolist())
        
        if not all_returns:
            return -10, 10
        
        # Use percentiles to set Y-axis range, avoiding extreme values
        lower_bound = np.percentile(all_returns, percentile_clip)
        upper_bound = np.percentile(all_returns, 100 - percentile_clip)
        
        # Add some margin
        margin = (upper_bound - lower_bound) * 0.1
        y_min = lower_bound - margin
        y_max = upper_bound + margin
        
        # Ensure zero axis is included
        if y_min > 0:
            y_min = min(y_min, -5)
        if y_max < 0:
            y_max = max(y_max, 5)
            
        return y_min, y_max
    
    def plot_industry(self, industry_name, industry_data, figsize=(15, 10)):
        """Plot cumulative return trend chart for a single industry"""
        
        # Create output directory
        output_dir = "industry_stock_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create chart
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set colors
        colors = plt.cm.Set3(np.linspace(0, 1, len(industry_data)))
        
        # Calculate appropriate Y-axis range
        y_min, y_max = self._calculate_y_limits(industry_data)
        
        # Plot each stock's trend
        valid_stocks = 0
        for i, (stock_name, df) in enumerate(industry_data.items()):
            if 'Cumulative Return' not in df.columns:
                continue
                
            # Plot cumulative return curve
            ax.plot(df.index, df['Cumulative Return'], 
                   color=colors[i % len(colors)], 
                   linewidth=2, 
                   alpha=0.8,
                   label=stock_name)
            
            valid_stocks += 1
        
        if valid_stocks == 0:
            print(f"  ❌ {industry_name} industry has no valid data")
            plt.close()
            return
        
        print(f"  ✓ {industry_name} industry: {valid_stocks} stocks")
        
        # Get time range information
        sample_df = next(iter(industry_data.values()))
        start_date = sample_df.index[0].strftime("%Y-%m-%d")
        end_date = sample_df.index[-1].strftime("%Y-%m-%d")
        
        # Set chart style
        ax.set_title(f'{industry_name} Industry Cumulative Return Trends\n'
                    f'Period: {start_date} to {end_date} ({valid_stocks} stocks)',
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        
        # Set Y-axis range
        ax.set_ylim(y_min, y_max)
        
        # Add horizontal zero line
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        
        # Set grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set legend - adjust based on number of stocks
        if valid_stocks <= 8:
            # 8 or fewer stocks, show on the right side
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        elif valid_stocks <= 12:
            # 9-12 stocks, show at bottom with 2 columns
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                     fontsize=9, ncol=2,
                     frameon=True, fancybox=True, shadow=True)
        elif valid_stocks <= 18:
            # 13-18 stocks, show at bottom with 3 columns
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', 
                     fontsize=8, ncol=3,
                     frameon=True, fancybox=True, shadow=True)
        else:
            # More than 18 stocks, don't show legend, display count on chart
            ax.text(0.02, 0.98, f'Contains {valid_stocks} stocks (legend hidden)', 
                   transform=ax.transAxes, fontsize=11,
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Beautify axes
        ax.tick_params(axis='both', which='major', labelsize=10)
        
        # Auto-adjust date display
        fig.autofmt_xdate()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save image
        # Handle special characters in filename
        safe_industry_name = industry_name.replace('/', '_').replace('\\', '_')
        filename = f"{safe_industry_name}_Industry_Trends.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        
        print(f"  💾 Saved image: {filepath}")
        
        # Show chart
        plt.show()
        
        # Close chart to free memory
        plt.close()
    
    def plot_all_industries(self, show_progress=True):
        """Plot trend charts for all industries"""
        if not self.stock_data:
            print("No data to plot, please run load_stock_data() first")
            return 0
        
        industry_groups = self.group_by_industry()
        
        print(f"\n📊 Industry grouping results:")
        for industry, stock_dict in industry_groups.items():
            print(f"  {industry}: {len(stock_dict)} stocks")
        
        total_industries = len(industry_groups)
        
        for i, (industry_name, industry_data) in enumerate(industry_groups.items()):
            if not industry_data:
                continue
            
            print(f"\n🎯 Plotting {industry_name} industry ({i+1}/{total_industries})")
            
            self.plot_industry(industry_name, industry_data)
            
            if show_progress and i < total_industries - 1:
                # Ask if continue (optional)
                response = input(f"Press Enter to continue to next industry, or enter 'q' to quit: ").strip().lower()
                if response == 'q':
                    print("User chose to exit")
                    return i + 1
        
        return total_industries
    
    def plot_single_industry(self, industry_name):
        """Plot trend chart for specified industry"""
        if not self.stock_data:
            print("No data to plot, please run load_stock_data() first")
            return
        
        industry_groups = self.group_by_industry()
        
        if industry_name not in industry_groups:
            print(f"Industry not found: {industry_name}")
            print(f"Available industries: {list(industry_groups.keys())}")
            return
        
        industry_data = industry_groups[industry_name]
        print(f"Plotting {industry_name} industry chart...")
        self.plot_industry(industry_name, industry_data)
    
    def list_industries_info(self):
        """List information for all industries"""
        if not self.stock_data:
            print("No data, please run load_stock_data() first")
            return
        
        industry_groups = self.group_by_industry()
        
        print(f"\n📋 Industry Information Overview:")
        print("=" * 60)
        
        for industry_name, stock_dict in industry_groups.items():
            if not stock_dict:
                continue
            
            print(f"\n📊 {industry_name}:")
            print(f"  Number of stocks: {len(stock_dict)}")
            
            # Show all stock names
            stock_names = list(stock_dict.keys())
            if len(stock_names) <= 10:
                print(f"  Stock list: {', '.join(stock_names)}")
            else:
                # If too many stocks, display in multiple lines
                print(f"  Stock list: ")
                for i in range(0, len(stock_names), 6):
                    batch = stock_names[i:i+6]
                    print(f"    {', '.join(batch)}")
    
    def generate_industry_summary(self):
        """Generate industry summary statistics"""
        if not self.stock_data:
            print("No data, please run load_stock_data() first")
            return
        
        industry_groups = self.group_by_industry()
        
        print(f"\n📈 Industry Summary Statistics:")
        print("=" * 80)
        print(f"{'Industry Name':<30} {'Stock Count':<12} {'Latest Return Range':<25} {'Average Return':<15}")
        print("-" * 80)
        
        for industry_name, stock_dict in industry_groups.items():
            if not stock_dict:
                continue
            
            returns = []
            for stock_name, df in stock_dict.items():
                if 'Cumulative Return' in df.columns and len(df) > 0:
                    latest_return = df['Cumulative Return'].iloc[-1]
                    if not pd.isna(latest_return):
                        returns.append(latest_return)
            
            if returns:
                min_return = min(returns)
                max_return = max(returns)
                avg_return = np.mean(returns)
                
                range_str = f"{min_return:+6.1f}% ~ {max_return:+6.1f}%"
                avg_str = f"{avg_return:+6.1f}%"
                
                print(f"{industry_name:<30} {len(stock_dict):<12} {range_str:<25} {avg_str:<15}")
            else:
                print(f"{industry_name:<30} {len(stock_dict):<12} {'No valid data':<25} {'N/A':<15}")

def main():
    """Main function"""
    print("🚀 Starting industry-based stock data analysis")
    print("=" * 50)
    
    # Configuration parameters
    data_folder = 'data/processed/train'  # Modify to your training data path
    
    if not os.path.exists(data_folder):
        print(f"❌ Data folder does not exist: {data_folder}")
        print("Please confirm the data path is correct")
        return
    
    # Create analyzer
    analyzer = IndustryStockAnalyzer(data_folder)
    
    # Load data
    file_count = analyzer.load_stock_data()
    
    if file_count == 0:
        print("❌ No stock data loaded")
        return
    
    # Show industry information
    analyzer.list_industries_info()
    
    # Generate summary statistics
    analyzer.generate_industry_summary()
    
    print(f"\nSelect operation mode:")
    print(f"1. Plot all industry charts (all at once)")
    print(f"2. Plot industry charts progressively (controllable progress)")
    print(f"3. Plot specific industry chart")
    print(f"4. Show industry information and statistics only")
    
    choice = input("Please select (1/2/3/4): ").strip()
    
    if choice == '1':
        # Plot all industries at once
        total_industries = analyzer.plot_all_industries(show_progress=False)
        print(f"\n✅ All industry charts completed! Generated {total_industries} industry charts in total")
        
    elif choice == '2':
        # Plot progressively, controllable progress
        total_industries = analyzer.plot_all_industries(show_progress=True)
        print(f"\n✅ Chart generation completed! Generated {total_industries} industry charts in total")
        
    elif choice == '3':
        # Plot specific industry
        industry_groups = analyzer.group_by_industry()
        print(f"\nAvailable industries:")
        industry_list = list(industry_groups.keys())
        for i, industry in enumerate(industry_list):
            print(f"  {i + 1}. {industry} ({len(industry_groups[industry])} stocks)")
        
        try:
            industry_choice = int(input("Please select industry (enter number): ")) - 1
            selected_industry = industry_list[industry_choice]
            
            analyzer.plot_single_industry(selected_industry)
            
        except (ValueError, IndexError) as e:
            print(f"Input error: {e}")
    
    elif choice == '4':
        print("✅ Industry information and statistics display completed")
    
    else:
        print("Invalid selection")
    
    if choice in ['1', '2', '3']:
        print(f"\n📁 Images saved to: industry_stock_plots/")

if __name__ == "__main__":
    main()
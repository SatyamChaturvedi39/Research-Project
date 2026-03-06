"""
Download the latest 2025-26 player statistics from Basketball-Reference
and save them to the data/raw/ directory without processing.
"""
import urllib.request
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

# Target URL for the 2025-26 season per-game stats
BBR_URL = "https://www.basketball-reference.com/leagues/NBA_2026_per_game.html"
RAW_DATA_DIR = "data/raw"

def main():
    print(f"Fetching latest player stats from Basketball-Reference...")
    print(f"URL: {BBR_URL}")

    # Ensure the raw data directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Use a descriptive filename with today's date
    today = datetime.now().strftime("%Y%m%d")
    output_filename = f"bbr_per_game_2025_26_{today}.html"
    output_path = os.path.join(RAW_DATA_DIR, output_filename)

    # Basketball-Reference requires a User-Agent to prevent blocking
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    req = urllib.request.Request(BBR_URL, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html_content = resp.read().decode('utf-8')
        
        # Save the raw HTML table content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Verify we got actual data (the table id is typically 'per_game_stats')
        if 'id="per_game_stats"' in html_content:
            print(f"\n✓ Successfully downloaded raw stats.")
            print(f"✓ Saved to: {output_path}")
            print(f"✓ File size: {len(html_content) / 1024:.1f} KB")
            print("\nNote: This is the raw HTML. You will need a separate script (like pandas.read_html) to parse this into a CSV.")
        else:
            print(f"\n⚠ Downloaded file, but could not find the expected 'per_game_stats' table in the HTML.")
            print(f"Saved to: {output_path} for inspection.")
            
    except urllib.error.HTTPError as e:
        print(f"\n❌ HTTP Error {e.code}: {e.reason}")
        if e.code == 429:
            print("Basketball-Reference rate limit hit. Try again later.")
    except Exception as e:
        print(f"\n❌ Failed to download: {str(e)}")

if __name__ == "__main__":
    main()

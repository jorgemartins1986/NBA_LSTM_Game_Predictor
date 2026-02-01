"""
Setup Odds API Integration
===========================

This script helps you set up and test the Odds API integration.

Steps:
1. Get a free API key from https://the-odds-api.com/
2. Set the ODDS_API_KEY environment variable
3. Run this script to test the connection

Usage:
    # Set API key (PowerShell)
    $env:ODDS_API_KEY = "your_api_key_here"
    python scripts/setup_odds_api.py
    
    # Or pass key directly
    python scripts/setup_odds_api.py --key YOUR_API_KEY
"""

import argparse
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.odds_api import OddsAPIClient


def main():
    parser = argparse.ArgumentParser(description='Test Odds API Integration')
    parser.add_argument('--key', type=str, help='API key (or set ODDS_API_KEY env var)')
    parser.add_argument('--save', action='store_true', help='Save key to .env file')
    args = parser.parse_args()
    
    print("="*60)
    print("ODDS API SETUP & TEST")
    print("="*60)
    
    # Get API key
    api_key = args.key or os.environ.get('ODDS_API_KEY')
    
    if not api_key:
        print("\n❌ No API key found!")
        print("\nTo get an API key:")
        print("  1. Go to https://the-odds-api.com/")
        print("  2. Sign up for free (500 credits/month)")
        print("  3. Copy your API key")
        print("\nTo use the key:")
        print('  PowerShell: $env:ODDS_API_KEY = "your_key_here"')
        print('  CMD: set ODDS_API_KEY=your_key_here')
        print('  Or run: python scripts/setup_odds_api.py --key YOUR_KEY')
        return
    
    print(f"\n✓ API key found: {api_key[:8]}...{api_key[-4:]}")
    
    # Save to .env if requested
    if args.save:
        env_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        with open(env_file, 'a') as f:
            f.write(f'\nODDS_API_KEY={api_key}\n')
        print(f"✓ Saved to {env_file}")
    
    # Test API connection
    print("\n📡 Testing API connection...")
    client = OddsAPIClient(api_key=api_key)
    
    import requests
    response = requests.get(
        f"{client.BASE_URL}/sports",
        params={'apiKey': api_key}
    )
    
    if response.status_code == 200:
        sports = response.json()
        nba = next((s for s in sports if s['key'] == 'basketball_nba'), None)
        
        print("✅ API connection successful!")
        print(f"   NBA active: {nba.get('active', False) if nba else 'Not found'}")
        print(f"\n📊 Quota Status:")
        print(f"   Remaining: {response.headers.get('x-requests-remaining', 'N/A')}")
        print(f"   Used: {response.headers.get('x-requests-used', 'N/A')}")
        
        # Try getting live odds (costs 1 credit)
        print("\n🏀 Fetching live NBA odds (costs 1 credit)...")
        odds = client.get_live_odds(regions='eu', odds_format='decimal')
        
        if odds:
            print(f"✅ Found {len(odds)} upcoming games with odds:\n")
            for game in odds[:3]:  # Show first 3
                features = client.extract_odds_features(game)
                home = game.get('home_team', 'Unknown')
                away = game.get('away_team', 'Unknown')
                
                print(f"   📍 {away} @ {home}")
                
                home_odds = features.get('HOME_AVG_ODDS', 0)
                away_odds = features.get('AWAY_AVG_ODDS', 0)
                home_prob = features.get('HOME_IMPLIED_PROB', 0)
                away_prob = features.get('AWAY_IMPLIED_PROB', 0)
                spread = features.get('HOME_ODDS_SPREAD', 0)
                
                if home_odds and away_odds:
                    print(f"      Home: {home_odds:.2f} (implied: {home_prob*100:.1f}%)")
                    print(f"      Away: {away_odds:.2f} (implied: {away_prob*100:.1f}%)")
                    print(f"      Spread: {spread:.3f} ({features.get('BOOKMAKER_COUNT', 0)} bookmakers)")
                print()
        else:
            print("   No games found (off-season or no upcoming games)")
            
    elif response.status_code == 401:
        print("❌ Invalid API key!")
    elif response.status_code == 429:
        print("⚠️ Rate limit exceeded. Wait a minute and try again.")
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("""
To use odds in predictions:
  1. Set the ODDS_API_KEY environment variable
  2. Run: python main.py predict
  
The predictions will automatically:
  - Fetch live bookmaker odds
  - Display odds alongside model predictions
  - Show "value" when model disagrees with market

To enrich training data with historical odds:
  - This costs 10 credits per day of data
  - With 500 free credits: ~50 days of historical data
  - See src/odds_api.py OddsDataEnricher class
""")


if __name__ == '__main__':
    main()

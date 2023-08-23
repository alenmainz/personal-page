import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
from datetime import datetime, timedelta
import os
import csv
import re
import time


# NBA Dictionary
nba_teams = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

# Altitude dictionary
nba_cities_altitudes = {
    "ATL": 1050,    # Altitude in feet
    "BOS": 141,      # Altitude in feet
    "BRK": 30,     # Altitude in feet
    "CHO": 751,   # Altitude in feet
    "CHI": 594,     # Altitude in feet
    "CLE": 653,   # Altitude in feet
    "DAL": 430,      # Altitude in feet
    "DEN": 5280,     # Altitude in feet
    "DET": 600,     # Altitude in feet
    "GSW": 13, # Altitude in feet (San Francisco)
    "HOU": 43,      # Altitude in feet
    "IND" : 715,
    "LAC": 233,          # Altitude in feet (Los Angeles)
    "LAL": 233, # Altitude in feet (Los Angeles)
    "MEM": 337,     # Altitude in feet
    "MIA": 6,         # Altitude in feet
    "MIL": 617,   # Altitude in feet
    "MIN": 830,   # Altitude in feet (Minneapolis)
    "NOP": 7,   # Altitude in feet
    "NYK": 33,     # Altitude in feet
    "OKC": 1200, # Altitude in feet
    "ORL": 82,      # Altitude in feet
    "PHI": 39, # Altitude in feet
    "PHX": 1086,    # Altitude in feet
    "POR": 43,     # Altitude in feet
    "SAC": 30,   # Altitude in feet
    "SAS": 650, # Altitude in feet
    "TOR": 249,     # Altitude in feet
    "UTA": 4226,       # Altitude in feet
    "WAS": 72   # Altitude in feet
}

column_names = ["home_team", 'away_team', "date", "team_of_stats", "winning_team", "home_game" ,
                'total_points', 'total_assists', 'total_rebounds', 'total_steals', 'total_blocks', 'total_turnovers',
                'team_true_shooting', 'team_effective_shooting', 'team_rebound_rate', 'team_assist_rate',
                'team_steal_rate','team_block_rate','team_turnover_rate','team_off_rtg',
                'team_def_rtg' ,'win']

os.chdir('/Users/bosnianthundaa/Documents/Basketball Prediction Project/Game Table URLs')
team_abbreviations = ["ATL", "BOS", "BRK", "CHO", "CHI", "CLE", "DAL", "DEN", "DET", "GSW",
                      "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NOP", "NYK",
                      "OKC", "ORL", "PHI", "PHO", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]


years_of_interest = ['2017', '2018','2019','2020','2021','2022']

team_abbreviations = ["HOU", "IND", "LAC", "LAL"]


team_abbreviations = ["SAC"]

years_of_interest = ['2019','2020','2021','2022']
base_url = "https://www.basketball-reference.com{}"
final_data = pd.DataFrame(columns = column_names)
headers = {
    "User-Agent" : "MyScraper (contact: babagansounds@gmail.com"}
# Specify the URL for the TEAM and YEAR
for team in team_abbreviations:
    for year_of_interest in years_of_interest:
        time.sleep(10)
        team_year_CSV_name = f'{team}_game_urls_{year_of_interest}.csv'
        team_year_CSV = pd.read_csv(team_year_CSV_name)
        print(team_year_CSV_name)
        team_year_URL_list = team_year_CSV['Game URLs'].tolist()
        for boxscore in team_year_URL_list:
            url = base_url.format(boxscore)
            print(url)
           
           
                       
            # Getting home and away team names
            time.sleep(6)
            response = requests.get(url, headers = headers)
            print(response)
            # print(response.headers["Retry-After"])
            html = response.content
            soup = BeautifulSoup(html, "html.parser")
           
           
            ##############################################################################
            # Get date of match
            date_start = url.find('/boxscores/') + len('/boxscores/')
            date_end = date_start + 8
            date_string = url[date_start:date_end]
           
            year = date_string[:4]
            month = date_string[4:6]
            day = date_string[6:]
           
            formatted_date = f"{year}-{month}-{day}"
            print("Extracted Date:", formatted_date)
           
            # Extract team names from the "strong" tags with specific attributes
            team_name_elements = soup.find_all("a", href=re.compile(r'/teams/[A-Z]{3}/\d+.html'))
            team_name_elements = team_name_elements[:2]
           
            # Extract and print the team names
            team_names = [element.get_text() for element in team_name_elements]
            print("Team Names:", team_names)
           
            home_team = team_names[0]
            away_team = team_names[1]
           
            # Extract home team
            # home_team = soup.find("strong", {"itemprop": "performer"}).text
            home_team_abb = nba_teams.get(home_team)
            # Extract away team
            # away_team = soup.find("strong", {"itemprop": "performer", "class": "loser"}).text
            away_team_abb = nba_teams.get(away_team)
           
           
            ##############################################################################
           
            ##############################################################################
            # Send an HTTP GET request to retrieve the HTML content
            # response = requests.get(url)
            html_content = response.text
           
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            # Find the table element that contains the basic Lakers' statistics
            basic_table = soup.find("table", id=f'box-{team}-game-basic')
           
               
            # Extract the basic statistics (points, assists, rebounds, turnovers)
            basic_team_stats_row = basic_table.find("tfoot")
           
            # if basic_team_stats_row is not None:
            #     # Use .prettify() to display the table's HTML structure
            #     # print(basic_team_stats_row.prettify())
            #     else:
            #         print("Table not found.")
               
            team_points = int(basic_team_stats_row.find("td", {"data-stat": "pts"}).text)
            team_assists = int(basic_team_stats_row.find("td", {"data-stat": "ast"}).text)
            team_rebounds = int(basic_team_stats_row.find("td", {"data-stat": "trb"}).text)
            team_steals = int(basic_team_stats_row.find("td", {"data-stat": "stl"}).text)
            team_blocks = int(basic_team_stats_row.find("td", {"data-stat": "blk"}).text)
            team_turnovers = int(basic_team_stats_row.find("td", {"data-stat": "tov"}).text)
            ##############################################################################
           
            ##############################################################################
            # Find the table element that contains the advanced Lakers' statistics
            advanced_table = soup.find("table", id=f'box-{team}-game-advanced')
               
            # Extract the specific statistics (points, assists, rebounds, turnovers)
            advanced_team_stats_row = advanced_table.find("tfoot")
           
            # if advanced_team_stats_row is not None:
            #     # Use .prettify() to display the table's HTML structure
            #     # print(advanced_team_stats_row.prettify())
            # else:
            #     print("Table not found.")
               
            team_trueshoot = float(advanced_team_stats_row.find("td", {"data-stat": "ts_pct"}).text)
            team_effshoot = float(advanced_team_stats_row.find("td", {"data-stat": "efg_pct"}).text)
            team_totalreb_rate = float(advanced_team_stats_row.find("td", {"data-stat": "trb_pct"}).text)
            team_totalass_rate = float(advanced_team_stats_row.find("td", {"data-stat": "ast_pct"}).text)
            team_totalstl_rate = float(advanced_team_stats_row.find("td", {"data-stat": "stl_pct"}).text)
            team_totalblk_rate = float(advanced_team_stats_row.find("td", {"data-stat": "blk_pct"}).text)
            team_totaltov_rate = float(advanced_team_stats_row.find("td", {"data-stat": "tov_pct"}).text)
            team_offrtg= float(advanced_team_stats_row.find("td", {"data-stat": "off_rtg"}).text)
            team_defrtg = float(advanced_team_stats_row.find("td", {"data-stat": "def_rtg"}).text)
            ##############################################################################
           
           
            ##############################################################################
            # Find the winner of the game
           
            # Find the final scores of the game
            # Find the final scores of the game using more specific parsing
            scorebox = soup.find("div", class_="scorebox")
            score_divs = scorebox.find_all("div", class_="score")
            score_divs
            scores = [score_div.get_text() for score_div in score_divs]
           
            print("Scores:", scores)
           
            home_team_scores = scores[0]
            away_team_scores = scores[1]
           
            # Determine the winner based on the scores
            if home_team_scores > away_team_scores:
                winner = "Home Team"
            elif away_team_scores > home_team_scores:
                winner = "Away Team"
            else:
                winner = "Tie"
           
            if winner == 'Home Team':
                winning_team = home_team_abb
            else:
                winning_team = away_team_abb
               
            print("Winner:", winning_team)

            if winning_team == team:
                win = 1
            else:
                win = 0          
            ##############################################################################
            if team == home_team_abb:
                playing_at_home = 1
            else:
                playing_at_home = 0
            ##############################################################################
            # Add all stats to a DataFrame
           
       
           
            # Replace these with your actual variables
            # final_stats_to_add = pd.DataFrame()
            final_stats_to_add = pd.DataFrame(columns=column_names)
            stats_to_add = [home_team_abb,away_team_abb, formatted_date, team, winning_team, playing_at_home, team_points,
                            team_assists, team_rebounds, team_steals, team_blocks, team_turnovers, team_trueshoot,
                            team_effshoot, team_totalreb_rate, team_totalass_rate, team_totalstl_rate, 
                            team_totalblk_rate, team_totaltov_rate, team_offrtg, team_defrtg, win]
            stats_to_add_df = pd.DataFrame([stats_to_add], columns=column_names)

            # Concatenate the new row DataFrame with the existing DataFrame
            final_data = pd.concat([final_data, stats_to_add_df], ignore_index=True)

            # final_stats_to_add = final_stats_to_add.append(stats_to_add, ignore_index=True)
            # final_stats_to_add = final_stats_to_add.append({
            #     "home_team": home_team_abb,
            #     "away_team": away_team_abb,
            #     "date": formatted_date,
            #     "team_of_stats" : team,
            #     "winning_team" : winning_team,
            #     # "location_of_game": [value3],
            #     "home_game": playing_at_home,
            #     # "altitude": [value2],
            #     # "back_to_back": [value3],
            #     "total_points": team_points,
            #     "total_assists": team_assists,
            #     "total_rebounds": team_rebounds,
            #     "total_steals": team_steals,
            #     "total_blocks": team_blocks,
            #     "total_turnovers": team_turnovers,
            #     "team_true_shooting": team_trueshoot,
            #     "team_effective_shooting": team_effshoot,
            #     "team_rebound_rate": team_totalreb_rate,
            #     "team_assist_rate": team_totalass_rate,
            #     "team_steal_rate": team_totalstl_rate,
            #     "team_block_rate": team_totalblk_rate,
            #     "team_turnover_rate": team_totaltov_rate,
            #     "team_off_rtg": team_offrtg,
            #     "team_def_rtg": team_defrtg,
            #     "win" : win
            #     # "wins_in_last_10": [value3],
            #     # "star_player_in": [value2]
            #    }, ignore_index=True)
            
            # final_data = final_data.append(final_stats_to_add, ignore_index=True)
            final_data.to_csv('all_game_logs.csv', index=False)
            ##############################################################################
            
    
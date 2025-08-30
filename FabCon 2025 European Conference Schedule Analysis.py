#!/usr/bin/env python
# coding: utf-8

# ## FabCon 2025 European Conference Schedule Analysis
# 
# New notebook

# # **Welcome to Vienna**
# 
# We’re here in the beautiful city of Vienna for **FabCon 2025**, where innovators, data enthusiasts, and technology leaders come together to share ideas, explore the latest trends, and collaborate on new solutions.  
# 
# To start our journey, we’ll dive into **web scraping the conference data** and exploring insights from the sessions and speakers.
# 
# 

# ## **To kick off our journey, here’s the **FabCon 2025 Vienna logo** — a small emblem of the excitement and learning that awaits us.**

# In[50]:


from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

logo_url = "https://www.sharepointeurope.com/nitropack_static/bIzdMVbKVbEcBjshCfCgJMJeENfwVVUk/assets/images/optimized/rev-a347ba2/www.sharepointeurope.com/wp-content/uploads/2025/05/EMFCC_Vienna25_Logo_Primary_White.png"

response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))

# Convert to RGBA to avoid transparency warnings
logo = logo.convert("RGBA")

canvas_width = logo.width * 5
canvas_height = logo.height
bg_color = "#014238"
bg = Image.new("RGB", (canvas_width, canvas_height), color=bg_color)

x_offset = (canvas_width - logo.width) // 2
y_offset = 0

bg.paste(logo, (x_offset, y_offset), mask=logo)

plt.figure(figsize=(15,4))
plt.imshow(bg)
plt.axis("off")
plt.title("FabCon 2025 Vienna")
plt.show()


# # FabCon 2025 Schedule Scraper
# 
# **Author:** Prachi Jain  
# **LinkedIn:** [https://www.linkedin.com/in/prachi-jain-490099127/](https://www.linkedin.com/in/prachi-jain-490099127/)  
# **Company Website:** [https://www.dreamitcs.com/](https://www.dreamitcs.com/)
# 
# This notebook scrapes the **FabCon 2025 schedule** from the SharePoint Europe website and prepares it for analysis.
# 
# Here’s what we do step by step:
# 
# 1. **Fetch the webpage** using `requests` and parse it with `BeautifulSoup`.
# 2. **Extract session details** including Day, Time, Code, Tag/Group, Title, Link, Level, Topic, and Speakers.
# 3. **Clean the data**:
#    - Fill missing Day and Time values
#    - Drop empty rows
#    - Merge speaker-only rows with their main sessions
#    - Add a sequential index
# 4. **Convert the DataFrame to Spark** for large-scale processing.
# 5. **Save the data** as a Delta table in the Fabric Lakehouse, ready for analysis or dashboards.
# 
# At the end, we’ll have a **clean, structured dataset** of all FabCon 2025 sessions and speakers, ready for exploration and insights.
# 

# In[51]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from pyspark.sql import SparkSession

# Start Spark session
spark = SparkSession.builder.appName("FabCon2025").getOrCreate()

# Fetch the webpage
url = "https://www.sharepointeurope.com/conference/schedule/2025-Fabric/"
headers = {"User-Agent": "Mozilla/5.0"}
response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.content, "html.parser")

all_sessions = []

# Loop through each day
day_blocks = soup.find_all("div", class_="tb-day")
for day_block in day_blocks:
    day_name = day_block.get("data-name", "Unknown Day")
    
    tables = day_block.find_all("table")
    for table in tables:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all("td")
            if not cols:
                continue

            # Initialize session fields
            time = code = tag = title = link = level = topic = speakers = ""
            
            # Get the first non-empty time
            for t_col in row.find_all("td", class_="time"):
                t_text = t_col.get_text(strip=True)
                if t_text:
                    time = t_text
                    break
            
            # Session code
            code_col = row.find("td", class_="code")
            if code_col:
                code = code_col.get_text(strip=True)
            
            # Tag/Group
            tag_col = row.find("td", class_="tag")
            if tag_col:
                tag = tag_col.get_text(strip=True)
            
            # Title and link (event-title or keynote-title)
            title_col = row.find("p", class_="event-title") or row.find("p", class_="keynote-title")
            if title_col:
                title = title_col.get_text(strip=True)
                link_tag = title_col.find("a", href=True)
                if link_tag:
                    link = link_tag["href"]
            else:
                # For rest-description rows
                rest_desc_col = row.find("td", class_="rest-description")
                if rest_desc_col:
                    title = rest_desc_col.get_text(strip=True)
            
            # Level and topic
            level_col = row.find("td", class_="level-cell")
            if level_col:
                level = level_col.get_text(strip=True)
            
            topic_col = row.find("td", class_="topic-cell")
            if topic_col:
                topic = topic_col.get_text(strip=True)
            
            # Speakers (meta rows)
            if "meta" in row.get("class", []):
                speaker_td = row.find("td", class_="light-bg")
                if speaker_td:
                    speakers = ", ".join([a.get_text(strip=True) for a in speaker_td.find_all("a")])
            
            # Skip empty rows
            if not any([time, code, tag, title, level, topic, speakers]):
                continue
            
            all_sessions.append({
                "Day": day_name,
                "Time": time,
                "Code": code,
                "Tag/Group": tag,
                "Title": title,
                "Link": link,
                "Level": level,
                "Topic": topic,
                "Speakers": speakers
            })

# Create initial DataFrame
df = pd.DataFrame(all_sessions)

# Fill down  Day/Time
df['Day'] = df['Day'].replace('', pd.NA).fillna(method='ffill')
df['Time'] = df['Time'].replace('', pd.NA).fillna(method='ffill')

# Drop completely empty sessions
df = df.dropna(subset=['Code', 'Title'], how='all')
df['Speakers'] = df['Speakers'].fillna('')

# Merge speaker-only rows with main sessions
merged_rows = []
previous_row = None
for _, row in df.iterrows():
    if row['Title']:
        if previous_row is not None:
            merged_rows.append(previous_row)
        previous_row = row.copy()
    else:
        if previous_row is not None and row['Speakers']:
            if previous_row['Speakers']:
                previous_row['Speakers'] += ", " + row['Speakers']
            else:
                previous_row['Speakers'] = row['Speakers']

# Add last session
if previous_row is not None:
    merged_rows.append(previous_row)

# Final cleaned DataFrame
final_df = pd.DataFrame(merged_rows)

# Add sequential index
final_df.insert(0, "Index", range(1, len(final_df) + 1))

# Convert to Spark DataFrame
dff = spark.createDataFrame(final_df)

# Display Spark DataFrame
display(dff.limit(5))

# Path
path = "abfss://MicrosoftPowerBI@onelake.dfs.fabric.microsoft.com/Lakehouse.Lakehouse/Tables/"

# Table name
delta_table_name = "EuropeFullProgramme"

# Save Spark DataFrame as Delta table in Fabric Lakehouse
dff.write.mode("overwrite").format("delta").save(f'{path}{delta_table_name}')

print(f"DataFrame successfully saved to table '{path}{delta_table_name}'")


# ## **Explore Insights from FabCon 2025**
# 
# Now that we have a **clean dataset** of all sessions and speakers, it's time to dive into insights.  
# 
# In this section, we will:
# 
# - **Number of Sessions per Day** – Shows how many sessions are scheduled each day.  
# - **Sessions per Topic (Top 10)** – Highlights the most common topics covered in the conference.  
# - **Most Active Speakers (Top 10)** – Identifies speakers who are presenting the most sessions.  
# - **Level Distribution** – Shows how sessions are distributed across skill/experience levels (Business, Technical, Advanced).  
# - **Sessions Over Time** – Shows session density by time of day to identify peak hours.
# 
# These insights will help us better understand the FabCon 2025 program and highlight the most important sessions and trends.
# 

# ###### **Number of Sessions per Day**
# ###### 
# ###### Shows which day is the busiest. 
# ###### Helps attendees prioritize their schedule.

# In[52]:


plt.figure(figsize=(10,6))
sns.countplot(data=pdf, x='Day', order=pdf['Day'].unique(), palette="viridis")
plt.title("Number of Sessions per Day")
plt.xlabel("Day")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ###### **Sessions per Topic (Top 10)**
# ###### 
# ###### Highlights the most common topics/themes at the conference.
# 

# In[53]:


top_topics = pdf['Topic'].value_counts().head(10).index
plt.figure(figsize=(12,6))
sns.countplot(data=pdf[pdf['Topic'].isin(top_topics)], y='Topic', order=top_topics, palette="coolwarm")
plt.title("Top 10 Topics by Number of Sessions")
plt.xlabel("Number of Sessions")
plt.ylabel("Topic")
plt.tight_layout()
plt.show()


# ###### **Most Active Speakers (Top 10)**
# ###### 
# ###### Shows speakers delivering multiple sessions
# 

# In[54]:


all_speakers = pdf['Speakers'].str.split(', ').explode()
top_speakers = all_speakers.value_counts().head(10).index

plt.figure(figsize=(12,6))
sns.countplot(y=all_speakers[all_speakers.isin(top_speakers)], order=top_speakers, palette="magma")
plt.title("Top 10 Speakers by Number of Sessions")
plt.xlabel("Number of Sessions")
plt.ylabel("Speaker")
plt.tight_layout()
plt.show()


# ###### **Level Distribution**
# ###### 
# ###### Shows how sessions are distributed across skill/experience levels (Business, Technical, Advanced).
# 

# In[55]:


plt.figure(figsize=(10,5))
sns.countplot(data=pdf, x='Level', palette="pastel", order=pdf['Level'].value_counts().index)
plt.title("Distribution of Sessions by Level")
plt.xlabel("Level")
plt.ylabel("Number of Sessions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ###### **Sessions Over Time**
# ###### Shows session density by time of day.
# ###### Useful to spot peak periods.
# 
# 

# In[56]:


# Convert Time to start hour
pdf['Start_Hour'] = pdf['Time'].str.extract(r'(\d{2}):')[0].astype(int)
plt.figure(figsize=(10,5))
sns.histplot(pdf['Start_Hour'], bins=range(8,19), kde=False, color='skyblue')
plt.title("Sessions Over Time")
plt.xlabel("Start Hour")
plt.ylabel("Number of Sessions")
plt.tight_layout()
plt.show()


# ## **Data Source**
# 
# The dataset used in this notebook comes from the **official FabCon 2025 conference website**:
# 
# [European Microsoft Fabric Community Conference 2025](https://www.sharepointeurope.com/european-microsoft-fabric-community-conference/)
# 
# I have **web scraped the schedule and session details** directly from this site to analyze the conference program, sessions, speakers, and topics.
# 

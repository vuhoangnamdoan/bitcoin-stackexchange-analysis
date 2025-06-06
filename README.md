# Bitcoin StackExchange Discussion Forum Analysis

A comprehensive data analysis project exploring the Bitcoin StackExchange dataset to understand user behavior, sentiment patterns, and discussion topics within the Bitcoin community.

## Contributing

This is an academic project for SIT220 Data Wrangling. For questions or suggestions:
- Student Name: **Vu Hoang Nam Doan**
- Student Number: s224021565
- Outlook: s224021565@deakin.edu.au
- Email: vuhoangnamdoan1605@gmail.com
- LinkedIn: [Vu Hoang Nam Doan](https://www.linkedin.com/in/vuhoangnamdoan/)
- Course: S379 - Bachelor of Data Science

However, suggestions for improvements are welcome:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## Project Overview

This project presents an in-depth analysis of the Bitcoin StackExchange dataset, a public Q&A platform for Bitcoin enthusiasts. The analysis covers approximately **32,000 questions** and **44,000 answers** to uncover insights into:

- User distribution and geographical patterns
- Sentiment analysis of posts and comments
- Topic modeling and trend analysis
- Community engagement patterns
- Tag co-occurrence networks

## Objectives

- Explore and understand the Bitcoin StackExchange user base
- Analyze sentiment expressed in discussions
- Identify key topics and themes in the community
- Map geographical distribution of users
- Understand community dynamics and engagement patterns

## Dataset

The dataset is sourced from Bitcoin StackExchange and includes:
- **Posts**: Questions and answers from community members
- **Comments**: User interactions and discussions
- **Users**: Profile information and user metadata
- **Tags**: Topic classifications for posts
- **Votes**: Community engagement metrics
- **Badges**: User achievement and reputation data

### Data Files
- Original XML format files (Posts.xml, Users.xml, Comments.xml, etc.)
- Converted CSV files for analysis
- Generated visualization outputs (HTML maps, charts)

## Methodology

### Data Processing
1. **XML to CSV Conversion**: Transform original StackExchange XML data into analyzable CSV format
2. **Data Cleaning**: Handle missing values, normalize text data
3. **Feature Engineering**: Extract relevant features for analysis

### Analysis Techniques
- **Exploratory Data Analysis (EDA)**: Statistical summaries and data visualization
- **Sentiment Analysis**: VADER sentiment analyzer with cryptocurrency-specific enhancements
- **Topic Modeling**: Latent Dirichlet Allocation (LDA) for topic discovery
- **Network Analysis**: Tag co-occurrence patterns
- **Geographical Analysis**: User location mapping and distribution

## Repository Structure

```
├── notebook.ipynb          # Main Jupyter notebook with complete analysis
├── notebook.py             # Python script version of the notebook
├── notebook.qmd            # Quarto document format
├── notebook.html           # HTML export of the analysis
├── notebook.pdf            # PDF export of the analysis
├── README.md               # This file
├── user_locations_map.html # Interactive user location map
├── Data Files:
│   ├── *.xml               # Original StackExchange XML dumps
│   │   ├── Posts.xml       # Questions and answers data
│   │   ├── Users.xml       # User profile information
│   │   ├── Comments.xml    # User comments and discussions
│   │   ├── Votes.xml       # Voting patterns and scores
│   │   ├── Tags.xml        # Tag definitions and usage
│   │   ├── Badges.xml      # User achievements and badges
│   │   ├── PostHistory.xml # Edit history of posts
│   │   └── PostLinks.xml   # Links between related posts
│   └── *.csv               # Processed CSV versions for analysis
│       ├── Posts.csv       # Cleaned posts data
│       ├── Users.csv       # Processed user information
│       ├── Comments.csv    # Processed comments
│       ├── Votes.csv       # Vote data
│       ├── Tags.csv        # Tag information
│       ├── Badges.csv      # Badge data
│       ├── PostHistory.csv # Post edit history
│       └── PostLinks.csv   # Post relationship data
```

### File Descriptions

#### Core Analysis Files
- **`notebook.ipynb`**: Interactive Jupyter notebook with complete analysis, visualizations, and narrative
- **`notebook.py`**: Python script version for command-line execution or automation
- **`notebook.qmd`**: Quarto document for reproducible research and publication-ready reports
- **`notebook.html`**: Static HTML version of the complete analysis with all outputs
- **`notebook.pdf`**: PDF export suitable for printing or academic submission

#### Data Files
- **XML Files**: Original StackExchange data dump files in XML format
- **CSV Files**: Preprocessed and cleaned data ready for analysis
- **`user_locations_map.html`**: Interactive geographical visualization of user locations

#### Key Data Tables
1. **Posts**: Contains questions, answers, creation dates, scores, and view counts
2. **Users**: User profiles, location, reputation, and registration information  
3. **Comments**: User interactions, replies, and discussions on posts
4. **Votes**: Community voting patterns and post scoring data
5. **Tags**: Topic classifications and tag usage statistics
6. **Badges**: User achievements, reputation milestones, and community recognition

## Getting Started

### Prerequisites

Install the required Python packages:

```bash
# Core data science libraries
pip install pandas numpy matplotlib seaborn scipy

# Natural language processing
pip install nltk scikit-learn gensim

# Network analysis and visualization
pip install networkx python-louvain folium

# Additional visualization tools
pip install wordcloud plotly
```

For NLTK, you may also need to download additional data:

```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
nltk.download('wordnet')
nltk.download('stopwords')
```

### Working with Different File Formats

#### 1. Jupyter Notebook (`notebook.ipynb`)
**Recommended for interactive analysis**

```bash
# Start Jupyter Lab
jupyter lab

# Or start Jupyter Notebook
jupyter notebook
```

- Open `notebook.ipynb` in your browser
- Run cells sequentially or all at once
- Modify code and see results immediately
- Best for exploration and visualization

#### 2. Python Script (`notebook.py`)
**For command-line execution or automation**

```bash
# Run the complete analysis
python notebook.py

# Or run with Python 3 explicitly
python3 notebook.py
```

- Contains the same analysis as the notebook
- Useful for batch processing or server environments
- Can be imported as a module in other Python scripts

#### 3. Quarto Document (`notebook.qmd`)
**For reproducible research and publication**

```bash
# Install Quarto first (https://quarto.org/docs/get-started/)
# Then render the document
quarto render notebook.qmd

# Render to specific formats
quarto render notebook.qmd --to html
quarto render notebook.qmd --to pdf
```

- Combines code, narrative, and results
- Generates professional reports
- Supports multiple output formats

#### 4. Pre-generated Reports
**For quick viewing without running code**

- **HTML Report** (`notebook.html`): Open directly in any web browser
- **PDF Report** (`notebook.pdf`): View with any PDF reader
- **Interactive Map** (`user_locations_map.html`): Open in browser for geographical analysis

### Key Dependencies

- **Data Processing**: pandas, numpy, scipy
- **NLP & Text Analysis**: nltk, scikit-learn, gensim
- **Visualization**: matplotlib, seaborn, plotly, wordcloud, folium
- **Network Analysis**: networkx, python-louvain
- **Machine Learning**: scikit-learn for topic modeling and classification

## Key Features

### 1. User Analysis
- Geographical distribution mapping
- User profile content analysis
- Activity patterns and engagement metrics

### 2. Sentiment Analysis
- VADER sentiment analysis with Bitcoin-specific lexicon
- Temporal sentiment trends
- Post vs. comment sentiment comparison

### 3. Topic Modeling
- LDA topic extraction and visualization
- Topic evolution over time
- Tag co-occurrence network analysis

### 4. Data Visualization
- Interactive geographical maps
- Word clouds and frequency analysis
- Network graphs for tag relationships
- Time series analysis of posting patterns

## Key Insights

The analysis reveals patterns in:
- Bitcoin community discussion topics
- User sentiment towards various Bitcoin-related subjects
- Geographical distribution of Bitcoin enthusiasts
- Evolution of discussion themes over time
- Community engagement and interaction patterns

## Data Privacy and Ethics

This analysis uses publicly available data from Bitcoin StackExchange where users share information openly. The study follows ethical practices by:
- Analyzing data in aggregated groups to respect user privacy
- Focusing on trends rather than individual identification
- Maintaining transparency in methodology and findings
- Recognizing the sensitive nature of cryptocurrency discussions

## Formats Available

This analysis is available in multiple formats:
- **Interactive Notebook**: `notebook.ipynb` - Full interactive analysis
- **Python Script**: `notebook.py` - Executable Python version
- **Quarto Document**: `notebook.qmd` - Reproducible research format
- **HTML Report**: `notebook.html` - Web-viewable static report
- **PDF Report**: `notebook.pdf` - Printable document format

## License

This project is for educational purposes as part of the SIT220 coursework at Deakin University.

---

*This analysis provides insights into the Bitcoin StackExchange community through comprehensive data science techniques, combining statistical analysis, natural language processing, and data visualization to understand cryptocurrency discussion patterns.*

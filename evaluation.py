import pandas as pd
import pingouin as pg

# 1. Load CSV files
human_df = pd.read_csv('coding.csv')
ai_df = pd.read_csv('fisclipt_scores.csv')

# 2. Normalize file_name in human_df to match ai_df.video
# Assuming human_df.file_name looks like 'AG0914_FIS_Time1_Jackson'
human_df['video'] = (
    human_df['file_name'].astype(str)
    .str.replace('Time', 'T')
    .str.replace('_FIS', '')
)
ai_df['video'] = (
    ai_df['video'].astype(str).str.replace('.mp4', ''))

print("Human videos sample:", human_df['video'].unique()[:10])
print("AI   videos sample:", ai_df['video'].unique()[:10])
print("Human count:", len(human_df['video'].unique()))
print("AI   count:", len(ai_df['video'].unique()))

# 3. Identify overlapping videos
common_videos = set(human_df['video']).intersection(ai_df['video'])
print(f"Found {len(common_videos)} overlapping videos")

# 4. Filter both DataFrames to common videos
h_sub = human_df[human_df['video'].isin(common_videos)].copy()
a_sub = ai_df[ai_df['video'].isin(common_videos)].copy()

# 5. Merge on 'video'
merged = pd.merge(
    h_sub,
    a_sub,
    on='video',
    suffixes=('_human', '_ai')
)

# 6. Compute ICC for each dimension
dimensions = [
    'verbal_fluency', 'hope_and_pe', 'persuasiveness',
    'emotional_expression', 'wau', 'empathy', 'abc', 'arrr'
]

icc_results = []
for dim in dimensions:
    # Prepare data for ICC: columns rater and scores
    df_icc = merged[['video', f'{dim}_human', f'{dim}_ai']]
    df_melt = df_icc.melt(id_vars='video', 
                           value_vars=[f'{dim}_human', f'{dim}_ai'],
                           var_name='rater', value_name='score')
    
    # Compute ICC(3,1)
    icc_result = pg.intraclass_corr(data=df_melt, targets="video", raters="rater", ratings="score")
    icc_result = icc_result.query("Type == 'ICC3'").iloc[0]
    icc_results.append({    
        'dimension': dim,
        'ICC_value': icc_result['ICC'],
        'CI95%': icc_result['CI95%'],
    })
    

    
icc_df = pd.DataFrame(icc_results)
print("\nICC results by dimension:")
print(icc_df)

# 7. Optionally, save the results
icc_df.to_csv('icc_results.csv', index=False)
print("Saved ICC results to icc_results.csv")

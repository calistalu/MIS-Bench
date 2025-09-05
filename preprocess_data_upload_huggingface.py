import pandas as pd
import os 
from datasets import Dataset, DatasetDict

# Load the CSV and Excel files
transcript = pd.read_csv('data/fis_transcript.csv')
coding = pd.read_csv('data/coding.csv')
excel_data = pd.ExcelFile('data/FIS_score_rubric.xlsx')
txt_file_path = 'data/CoT.txt'
# Define the directory containing your video files
video_directory = 'data/FIS996'
with open(txt_file_path, 'r') as file:
    content = file.read()


# Function to get the file path of the corresponding video
def get_video_path(video_name):
    video_file = f"{video_name}.mp4"
    video_path = os.path.join(video_directory, video_file)
    
    # Check if the video file exists
    if os.path.exists(video_path):
        return video_path
    else:
        return None  # Return None if the file is not found

def extract_score(df, video_name,fis_name):
    import re
    video_name = re.sub(r'_T([123])', r'_FIS_Time\1', video_name)
    matched_row = df[df["file_name"] == video_name]
    if not matched_row.empty:
        return matched_row[fis_name].iloc[0]
    else:
        raise ValueError(f"Video name {video_name} not found in the coding DataFrame.")
        return None

def extract_text(content, video, index):
    """
    Extracts the text corresponding to a specific video and index from the content.
    """
    video_start_index = content.find(video)
    if video_start_index == -1:
        return None  # Video name not found
    print(f"Video:{video},Video start index: {video_start_index}")
    # Find the position of index_name after video_name
    index_start_index = content.find(str(index)+'. ', video_start_index)
    index_start_index= content.find('\n', index_start_index)
    index_end_index = content.find(str(index+1)+'. ', video_start_index)
    extracted_text = content[index_start_index: index_end_index].strip()
    print(index_start_index,index_end_index)
    print(f"Extracted text for {video} index {index}: {extracted_text}")
    return extracted_text

# Define the mapping of FIS to sheet names
fis_mapping = {
    'verbal_fluency': 'VF',
    'hope_and_pe': 'Hope',
    'persuasiveness': 'PER',
    'emotional_expression': 'EE',
    'wau': 'WAU',
    'empathy': 'EMP',
    'abc': 'ABC',
    'arrr': 'ARRR'
}



# Initialize an empty DataFrame for the transformed data
transformed_data = []
count = 0
# Iterate over each row in the CSV
for _, row in transcript.iterrows():
    video = row['video']
    transcript = row['transcript']
    video_path = get_video_path(video)

    # Extract data from the corresponding Excel sheet for each FIS
    for index, (fis_name, sheet_name) in enumerate(fis_mapping.items()):
        fis_sheet = excel_data.parse(sheet_name, header=0)
        # Extract the relevant columns (assumes the criteria and descriptions are in specific columns)
        orig_criteria = fis_sheet.iloc[0, 2]
        score = extract_score(coding, video,fis_name)
        # Ensure score_descriptions is a pandas Series with default integer indexing
        score_descriptions = fis_sheet.iloc[1:6, 2].reset_index(drop=True)  # reset the index
        feedback = extract_text(content, video, index + 1)
        if feedback:
            count += 1
        # print(f"Feedback for {video} FIS {index + 1}: {feedback}")
        # Append the transformed data
        transformed_data.append({
            'video': video,
            'video_path': video_path,
            'orig_response': transcript,
            'orig_instruction': "You are a state‐of‐the‐art vision‐language model. You will be shown: 1) A short video clip capturing a clinician responding to a patient's challenge situation. 2) The full transcript of the clinician response. 3) the score rubric. Your task is to analyze how the clinician responds to the patient’s emotional and interpersonal cues. For each of the eight Facilitative Interpersonal Skills (FIS) categories below, assign a numerical rating with one decimal place from 1 (low) to 5 (high) following the detailed rubric.",
            'feedback': feedback,  # This can be updated later
            'score': score,  # Assuming score is a single value for now
            'orig_criteria': orig_criteria,
            'orig_score1_description': score_descriptions[0],  # Description for score 1
            'orig_score2_description': score_descriptions[1],  # Description for score 2
            'orig_score3_description': score_descriptions[2],  # Description for score 3
            'orig_score4_description': score_descriptions[3],  # Description for score 4
            'orig_score5_description': score_descriptions[4],  # Description for score 5
        })
    print(f"Processed video: {video}, total feedback entries so far: {count}")
print(f"Total feedback entries extracted: {count}")
# Convert the transformed data into a DataFrame
final_df = pd.DataFrame(transformed_data)

# # Save the final dataset as CSV
# final_df.to_csv('final_dataset.csv', index=False)
from sklearn.model_selection import train_test_split

# 你列出来的必须在 train 的视频
force_train_videos = [
    "KM0302_FIS_Time1_Jackson",
    "ZB0910_FIS_Time1_Jackson",
    "AK0721_FIS_Time2_Savannah",
    "BM0616_FIS_T2_Savannah",
    "BV0706_FIS_Time2_John",
    "CS0920_FIS_Time3_Bethany",
    "EH0202_FIs_Time3_Luke",
    "KM0302_FIS_Time3_Lauren",
    "RL0702_FIS_Time3_Sean",
    "AKF0715_FIS_Time1_Jessica",
    "BY1218_FIS_Time1_Jessica",
    "AK0721_FIS_Time2_Savannah",
    "BY1218_FIS_Time2_Bethany",
    "JK0328_FIS_Time1_Jessica",
    "LY1205_FIS_Time2_Savannah",
    "AK0721_FIS_Time3_Bethany",
    "VS0305_FIS_Time3_Luke",
    "AS1031_FIS_Time1_Jessica",
    "AB0511_FIS_Time3_Bethany",
    "EH0822_FIS_Time2_Jessica",
    "KM0302_FIS_Time1_Jackson",
    "SN0216_FIS_Time1_Luke",
    "MF0524_FIS_Time3_Bethany",
    "BM0616_FIS_Time2_Luke",
    "MS0315_FIS_Time2_Luke",
    "CS0920_FIS_Time3_John",
    "AS1031_FIS_Time1_Jackson",
    "JS1112_FIS_Time2_Luke",
    "SB0619_FIS_Time1_Jackson",
    "RD0924_FIS_Time3_Luke"
]
force_train_videos = set(force_train_videos)  # 用 set 提高效率

# 强制进入 train 的行
train_df_forced = final_df[final_df["video"].isin(force_train_videos)]

# 其他候选视频
remaining_videos = set(final_df["video"].unique()) - force_train_videos

# train-test split (按视频分，不拆开)
train_videos, test_videos = train_test_split(
    list(remaining_videos), test_size=0.1, random_state=42
)
print(f"Train videos: {len(train_videos)}, Test videos: {len(test_videos)}")
pd.DataFrame({"video": test_videos}).to_csv("test_videos.csv", index=False)
print("Test video names saved to test_videos.csv")

train_df_split = final_df[final_df["video"].isin(train_videos)]
test_df = final_df[final_df["video"].isin(test_videos)]

# 拼合强制训练集
train_df = pd.concat([train_df_forced, train_df_split], ignore_index=True)

print(f"Train set: {train_df.shape}, Test set: {test_df.shape}")
# 确保没有 pandas index 留下来

train_df["feedback"] = train_df["feedback"].astype(str)
test_df["feedback"] = test_df["feedback"].astype(str)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)# 确保没有 pandas index 留下来

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 合并成 DatasetDict
dataset_dict = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# 一次性 push
dataset_dict.push_to_hub("FIS-Full-Dataset")
print("Datasets pushed to the Hugging Face Hub as 'FIS-Full-Dataset'")
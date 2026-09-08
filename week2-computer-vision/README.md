
⚽ Football Team Color Detection

A computer vision project that processes football match footage and automatically identifies the dominant team colors of players in the video.

The project uses video processing and image analysis techniques to detect players, analyze their jersey colors, and generate an annotated output video showing the detected team colors.

📌 Project Overview

The goal of this project is to automatically distinguish football players based on the colors of their kits.

Given a football match video, the system processes the footage frame by frame and analyzes the players' clothing to determine their team color.

Workflow
Input Football Video
        │
        ▼
   Video Processing
        │
        ▼
   Player Detection
        │
        ▼
  Jersey/Color Analysis
        │
        ▼
 Team Color Classification
        │
        ▼
 Annotated Output Video

✨ Features
🎥 Processes football match videos frame by frame
🧍 Identifies football players in the footage
🎨 Analyzes player jersey colors
🔵🔴 Classifies players according to team colors
🖼️ Generates an annotated video
📊 Provides processing progress during execution
💾 Saves the final processed video automatically
🛠️ Technologies

The project is built around computer vision and video-processing techniques. Depending on the implementation, it can use technologies such as:

Python
OpenCV
NumPy
Machine Learning / Computer Vision models
Video processing with MP4
📂 Project Structure

A typical project structure looks like this:

player_team_color/
│
├── input/
│   └── football_match.mp4
│
├── output/
│   └── football_team_colors.mp4
│
├── src/
│   └── ...
│
├── requirements.txt
├── README.md
└── main.py


The exact file structure may vary depending on the implementation.

🚀 Getting Started
1. Clone the Repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd player_team_color

2. Create a Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate


macOS/Linux:

python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt


If a requirements.txt file has not been created yet, install the required computer vision packages used by your implementation.

For example:

pip install opencv-python numpy

▶️ Running the Project

Place the football match video in the project's input directory and run the main Python script:

python main.py


During execution, the program displays processing progress similar to:

Processed 6100/6390
Processed 6200/6390
Processed 6300/6390

============================================================
PROCESSING COMPLETE
============================================================


Once processing is complete, the generated video will be saved in the output directory.

📤 Output

The processed video is generated as:

output/football_team_colors.mp4


For example:

Output video:
C:\Users\HP\player_team_color\output\football_team_colors.mp4


The output video contains the results of the team-color analysis applied to the original football footage.

Example Processing Result
Input Video
     │
     ├── Player 1 → Team Color A
     ├── Player 2 → Team Color B
     ├── Player 3 → Team Color A
     └── Player 4 → Team Color B
             │
             ▼
     Annotated Output Video

📈 Processing Progress

The application reports the number of processed frames so that you can monitor long-running video-processing jobs.

For example:

Processed 100/6390
Processed 200/6390
...
Processed 6390/6390


This is particularly useful when working with high-resolution or long football videos, where processing can take some time.

🎯 Use Cases

This project can be used as a foundation for:

Football player tracking
Team identification
Sports analytics
Player segmentation
Tactical analysis
Computer vision research
Automated sports video analysis
Team-based player tracking
Highlight and match-analysis systems
🔮 Future Improvements

Several improvements could make the project more robust and accurate:

 Improve player detection accuracy
 Add automatic team-color clustering
 Support more than two teams/colors
 Improve detection under different lighting conditions
 Handle similar-colored jerseys
 Add player tracking across frames
 Detect referees separately
 Add team possession statistics
 Add player statistics
 Generate heat maps
 Add real-time video processing
 Build a web interface for uploading videos
 Optimize processing speed using GPU acceleration
⚠️ Limitations

Color-based team identification can be affected by:

Different camera angles
Shadows and lighting
Motion blur
Players being partially occluded
Similar jersey colors
Goalkeeper kits
Referee clothing
Advertising boards and background colors

For best results, the input video should have reasonably clear views of the players and their jerseys.

📊 Current Processing Result

The project successfully processed a football video containing:

Total frames: 6,390
Processed:    6,390
Status:       COMPLETE
Output:       football_team_colors.mp4
Size:         ~264 MB

🤝 Contributing

Contributions are welcome!

If you would like to improve the project:

Fork the repository.
Create a new branch.
git checkout -b feature/improved-team-detection

Make your changes.
Commit your changes.
git commit -m "Improve team color detection"

Push the branch.
git push origin feature/improved-team-detection

Open a Pull Request.
📜 License

This project is available under the license included in the repository.

If no license has been selected yet, consider adding an appropriate open-source license such as the MIT License.

👨‍💻 Author

Ademigoke Michael

Built as a computer vision project for football video analysis.

⭐ If you find this project useful, consider giving the repository a star!

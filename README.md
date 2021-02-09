# vr-user-behavior-analysis
Exploring user behavior in Virtual Reality through heat map visualization and behavior classification

## Repository Structure

### Data

Experiment data from Dr. Prakash and his team

  - `VideosData`: Videos (both user traces visualization and source videos)
      - `Videos`: Actual VR Videos
          - `Source`: Source videos
          - `SourceFrames`: Frames for each video
      - `Visualized`: User traces videos
          - `Traces`: Tracing user viewports on black background
          - `TracesOnVideo`: Tracing user viewports on video background
  - `UserTracesByVideo`: User traces data for each video
  - `Misc`: Miscellaneous scripts/questionnaires that might be helpful

### 20F

Key components/visualizations from [Fall 2020 ACM Research](https://github.com/ACM-Research/vr-viewport-analysis)

- `demo`: Various images that might help in understanding the data as well as the research topic
- `scripts`: Useful scripts for visualizing overlays, image processing, as well as video to frames conversion
